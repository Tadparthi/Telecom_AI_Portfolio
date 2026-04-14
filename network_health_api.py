from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import pandas as pd
import io
import uvicorn
from dotenv import load_dotenv
import os
import httpx

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI(
    title="Network Cell Health Monitor API",
    description="AI-powered 5G/LTE cell health scoring and anomaly detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "kpi_upload_dashboard.html")
    )

@app.get("/single")
def serve_single():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "noc_dashboard.html")
    )
# In-memory store for full cell data including trends
cell_store = {}

# ── Schemas ────────────────────────────────────────────────
class CellMeasurement(BaseModel):
    sinr_db:                  float
    cqi_mean:                 float
    bler_dl:                  float
    prb_util_dl:              float
    prb_util_ul:              float
    ho_success_rate:          float
    abnormal_release_ratio:   float
    rlf_count:                float
    dl_tput_user:             Optional[float] = None
    ul_tput_user:             Optional[float] = None
    cell_id:                  Optional[str]   = None
    timestamp:                Optional[str]   = None

class HealthPrediction(BaseModel):
    cell_id:            Optional[str]
    health_score:       float
    health_band:        str
    health_color:       str
    anomaly_label:      str
    anomaly_score:      float
    flags:              list
    recommended_action: str
    priority:           str
    confidence:         float
    kpi_detail:         list

# ── Thresholds ─────────────────────────────────────────────
THRESHOLDS = {
    'sinr_db':                {'label': 'SINR',               'unit': 'dB',   'threshold': 5,     'direction': 'below', 'weight': 3.0},
    'cqi_mean':               {'label': 'CQI Mean',           'unit': '',     'threshold': 6,     'direction': 'below', 'weight': 2.5},
    'bler_dl':                {'label': 'BLER DL',            'unit': '%',    'threshold': 10,    'direction': 'above', 'weight': 2.0},
    'rlf_count':              {'label': 'RLF Count',          'unit': '',     'threshold': 50000, 'direction': 'above', 'weight': 2.0},
    'abnormal_release_ratio': {'label': 'Abnormal Release',   'unit': '%',    'threshold': 15,    'direction': 'above', 'weight': 1.5},
    'prb_util_dl':            {'label': 'PRB Util DL',        'unit': '%',    'threshold': 80,    'direction': 'above', 'weight': 1.5},
    'ho_success_rate':        {'label': 'HO Success Rate',    'unit': '%',    'threshold': 95,    'direction': 'below', 'weight': 2.0},
    'dl_tput_user':           {'label': 'DL Throughput/User', 'unit': 'Mbps', 'threshold': None,  'direction': None,    'weight': 0},
    'prb_util_ul':            {'label': 'PRB Util UL',        'unit': '%',    'threshold': None,  'direction': None,    'weight': 0},
}

# ── Core functions ─────────────────────────────────────────
def compute_health_score(m: CellMeasurement):
    flags = []; score = 0.0; kpi_detail = []
    kpi_values = {
        'sinr_db': m.sinr_db, 'cqi_mean': m.cqi_mean,
        'bler_dl': m.bler_dl, 'rlf_count': m.rlf_count,
        'abnormal_release_ratio': m.abnormal_release_ratio,
        'prb_util_dl': m.prb_util_dl, 'ho_success_rate': m.ho_success_rate,
        'dl_tput_user': m.dl_tput_user or 0, 'prb_util_ul': m.prb_util_ul,
    }
    for kpi, val in kpi_values.items():
        t = THRESHOLDS[kpi]; flagged = False
        if t['threshold'] is not None:
            if t['direction'] == 'below' and val < t['threshold']:
                flagged = True
                flags.append("RLF spike" if kpi == 'rlf_count' else f"{t['label']} poor")
                score += t['weight']
            elif t['direction'] == 'above' and val > t['threshold']:
                flagged = True
                flags.append("RLF spike" if kpi == 'rlf_count' else f"{t['label']} high")
                score += t['weight']
        threshold_str = f"{'<' if t['direction']=='below' else '>'} {t['threshold']}{t['unit']}" if t['threshold'] is not None else 'Monitor only'
        kpi_detail.append({
            'kpi': t['label'], 'value': round(val, 2), 'unit': t['unit'],
            'threshold': t['threshold'], 'threshold_str': threshold_str,
            'direction': t['direction'], 'flagged': flagged, 'weight': t['weight']
        })
    return score, flags, kpi_detail

def score_to_band(score):
    if score == 0:     return "perfect",  "#2ecc71", "none",     "No action required — all KPIs within threshold", 0.99
    elif score <= 1.5: return "healthy",  "#27ae60", "low",      "Monitor — minor KPI flag detected", 0.95
    elif score <= 3.5: return "watch",    "#f39c12", "medium",   "Investigate — multiple KPI flags, schedule review", 0.90
    elif score <= 6.0: return "degraded", "#e67e22", "high",     "Urgent — significant degradation, field check recommended", 0.85
    else:              return "critical", "#e74c3c", "critical", "Critical — immediate field investigation required", 0.80

def compute_anomaly(m: CellMeasurement):
    score = 0.0; reasons = []
    if m.sinr_db > 8 and m.cqi_mean < 7:
        score -= 0.3; reasons.append("Good SINR but poor CQI — possible interference pattern")
    if m.prb_util_dl > 70 and (m.dl_tput_user or 0) < 5:
        score -= 0.4; reasons.append("High PRB but low throughput — congestion or scheduling issue")
    if m.sinr_db > 12 and m.abnormal_release_ratio > 10:
        score -= 0.5; reasons.append("Good SINR but high abnormal release — possible hardware issue")
    if m.rlf_count > 30000 and m.sinr_db > 10:
        score -= 0.4; reasons.append("High RLF with good SINR — possible backhaul or config issue")
    return ("anomalous" if score < -0.3 else "normal"), round(score, 3), reasons

def build_result(m: CellMeasurement):
    score, flags, kpi_detail = compute_health_score(m)
    band, color, priority, action, conf = score_to_band(score)
    anomaly_label, anomaly_score, anomaly_reasons = compute_anomaly(m)
    if anomaly_label == "anomalous" and band == "healthy":
        action = "Anomalous KPI pattern detected despite normal thresholds — investigate"
        priority = "medium"
    return {
        "cell_id": m.cell_id, "health_score": round(score, 2),
        "health_band": band, "health_color": color,
        "anomaly_label": anomaly_label, "anomaly_score": anomaly_score,
        "anomaly_reasons": anomaly_reasons, "flags": flags,
        "flag_count": len(flags), "recommended_action": action,
        "priority": priority, "confidence": conf, "kpi_detail": kpi_detail,
    }

# ── Column detection ───────────────────────────────────────
KPI_KEYWORDS = {
    'sinr_db':                ['sinr','snr','pusch_sinr','pdsch_sinr','dl_sinr','ul_sinr','avg_sinr','mean_sinr','sinr_db','nr_5062b','sinr for pusch in rank 1','sinr for pusch','sinr for pucch'],
    'cqi_mean':               ['cqi','channel_quality','cqi_mean','avg_cqi','mean_cqi','cqi_avg','cqi_dl','nr_5060b','nr_5061b','wideband cqi','average wideband cqi'],
    'bler_dl':                ['bler','block_error','bler_dl','dl_bler','initial_bler','residual_bler','bler_rate','nr_5054a','nr_5055a','initial bler in downlink','bler in downlink','residual block error'],
    'prb_util_dl':            ['prb_dl','dl_prb','prb_util_dl','dl_prb_util','prb_utilization_dl','resource_block_dl','prb_usage_dl','dl_rb_util','prb_dl_util','dl_prb_utilization','prb_util','nr_5114a','prb util pdsch','pdsch prb','prb utilization pdsch'],
    'prb_util_ul':            ['prb_ul','ul_prb','prb_util_ul','ul_prb_util','prb_utilization_ul','resource_block_ul','prb_usage_ul','ul_rb_util','prb_ul_util','ul_prb_utilization','nr_5115a','prb util pusch','pusch prb','prb utilization pusch'],
    'ho_success_rate':        ['ho_success','handover_success','ho_sr','mobility_success','handoff_success','intra_ho_success','inter_ho_success','ho_succ_rate','handover_sr','ho_success_rate','mobility_sr','ho_performance','nr_5042b','nr_5034a','nr_5049a','intra frequency handover total success','handover total success ratio','intra gnb intra frequency handover'],
    'abnormal_release_ratio': ['abnormal_release','abnormal_rel','drop_rate','call_drop','rlc_drop','conn_drop','connection_drop','radio_link_failure','rrc_drop','abnormal_rel_ratio','drop_ratio','nr_5025a','nr_5026a','nr_5032b','sgnb triggered abnormal release ratio','abnormal release ratio excluding','ratio of ue releases due to abnormal','abnormal releases'],
    'rlf_count':              ['rlf','radio_link_fail','radio_link_failure','rlf_count','rlf_total','link_failure','rlf_num','num_rlf','rlf_detected','nr_5036e','radio link failures','rlf detected','nsa radio link failures'],
    'dl_tput_user':           ['tput','throughput','dl_tput','dl_throughput','user_tput','avg_tput','mean_tput','dl_user_tput','downlink_throughput','cell_tput_dl','average_throughput_dl','dl_cell_tput','tput_dl','nr_5100b','mac layer user throughput in downlink','average mac layer user throughput','user throughput in downlink'],
    'cell_id':                ['cell_id','cellid','cell_name','cellname','site_id','enodeb','gnodeb','cell','node_id','cell_identity','eci','cell_label','eutrancell','nrcell','object','site','gnb'],
}

def auto_detect_columns(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapped = {}; match_log = []
    for kpi, keywords in KPI_KEYWORDS.items():
        found = False
        for keyword in keywords:
            for col_lower, col_orig in cols_lower.items():
                if keyword in col_lower:
                    mapped[kpi] = col_orig
                    match_log.append(f"{kpi} → '{col_orig}'")
                    found = True; break
            if found: break
    required  = ['sinr_db','cqi_mean','bler_dl','prb_util_dl','prb_util_ul','ho_success_rate','abnormal_release_ratio','rlf_count']
    unmatched = [k for k in required if k not in mapped]
    return mapped, match_log, unmatched

def detect_time_column(df):
    for col in df.columns:
        if any(k in col.lower() for k in ['period','time','date']):
            for fmt in ['%m/%d/%Y','%d/%m/%Y','%Y-%m-%d','%Y/%m/%d']:
                try:
                    parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if parsed.notna().sum() > len(df) * 0.5:
                        return col, parsed
                except: pass
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    return col, parsed
            except: pass
    return None, None

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Network Cell Health Monitor API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
@app.post("/chat")
async def chat_proxy(request: dict):
    """
    Proxy OpenAI calls through the backend.
    Key stays on server — never exposed to browser.
    """
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured on server"}

    messages  = request.get('messages', [])
    max_tokens = request.get('max_tokens', 800)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type':  'application/json'
                },
                json={
                    'model':       'gpt-4o',
                    'messages':    messages,
                    'max_tokens':  max_tokens,
                    'temperature': 0.3
                }
            )
            return response.json()

    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}
@app.post("/predict", response_model=HealthPrediction)
def predict_health(measurement: CellMeasurement):
    return HealthPrediction(**build_result(measurement))

@app.post("/predict/batch")
def predict_batch(measurements: list[CellMeasurement]):
    results = [build_result(m) for m in measurements]
    results.sort(key=lambda x: x['health_score'], reverse=True)
    bands = [r['health_band'] for r in results]
    return {
        "total_cells": len(results),
        "critical": bands.count('critical'), "degraded": bands.count('degraded'),
        "watch": bands.count('watch'), "healthy": bands.count('healthy'),
        "perfect": bands.count('perfect'),
        "anomalous": sum(1 for r in results if r['anomaly_label']=='anomalous'),
        "predictions": results
    }

@app.get("/cell/{cell_id}")
def get_cell_detail(cell_id: str):
    """
    Fetch full detail including 14-day trend for a single cell.
    Only available after a file has been uploaded.
    """
    if cell_id in cell_store:
        return cell_store[cell_id]
    return {"error": f"Cell {cell_id} not found. Upload a file first."}

@app.post("/predict/upload")
async def predict_from_file(file: UploadFile = File(...)):
    global cell_store

    contents = await file.read()

    # ── Load file ──────────────────────────────────────────
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx','.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Only CSV or Excel files supported"}
    except Exception as e:
        return {"error": f"Could not read file: {str(e)}"}

    if len(df) == 0:
        return {"error": "File is empty"}

    # ── Detect time column ─────────────────────────────────
    time_col, time_series = detect_time_column(df)

    # ── Detect KPI columns ─────────────────────────────────
    mapped, match_log, unmatched = auto_detect_columns(df)

    required_strict = ['sinr_db','cqi_mean','bler_dl','prb_util_dl',
                       'prb_util_ul','ho_success_rate','abnormal_release_ratio']
    if unmatched:
        strict_missing = [k for k in unmatched if k in required_strict]
        if strict_missing:
            return {"error": f"Could not find columns for: {strict_missing}",
                    "your_columns": list(df.columns), "matched_so_far": match_log}

    # ── Build normalized dataframe ─────────────────────────
    df_norm = pd.DataFrame()
    for kpi, col in mapped.items():
        df_norm[kpi] = pd.to_numeric(df[col], errors='coerce').values

    if 'rlf_count' not in df_norm.columns or df_norm['rlf_count'].isna().all():
        df_norm['rlf_count'] = 0.0
        match_log.append("rlf_count → defaulted to 0")

    if 'cell_id' in mapped:
        df_norm['cell_id'] = df[mapped['cell_id']].astype(str).values
    else:
        df_norm['cell_id'] = [f"Cell_{i:04d}" for i in range(len(df))]

    if 'dl_tput_user' not in df_norm.columns:
        df_norm['dl_tput_user'] = 10.0

    # Timestamp — .values strips index to prevent alignment issues
    df_norm['_timestamp'] = time_series.values if time_series is not None else pd.NaT

    # ── Clean ──────────────────────────────────────────────
    df_norm  = df_norm.dropna(subset=required_strict).reset_index(drop=True)
    before   = len(df_norm)
    df_norm  = df_norm[df_norm['cqi_mean'] > 0].reset_index(drop=True)
    filtered = before - len(df_norm)

    if len(df_norm) == 0:
        return {"error": "No valid rows after cleaning."}

    # ── Date setup ─────────────────────────────────────────
    has_time       = df_norm['_timestamp'].notna().any()
    unique_periods = int(df_norm['_timestamp'].nunique()) if has_time else 1
    all_dates      = sorted(df_norm['_timestamp'].dropna().unique()) if has_time else []
    date_labels    = [pd.Timestamp(d).strftime('%m/%d') for d in all_dates]

    trend_kpis = ['sinr_db','cqi_mean','bler_dl','prb_util_dl',
                  'prb_util_ul','ho_success_rate','abnormal_release_ratio','dl_tput_user']
    good_high  = {'sinr_db','cqi_mean','ho_success_rate','dl_tput_user'}

    # ── Latest row per cell ────────────────────────────────
    if has_time:
        latest_df = df_norm.sort_values('_timestamp').groupby('cell_id').last().reset_index()
    else:
        latest_df = df_norm.groupby('cell_id').last().reset_index()

    # ── Pre-compute trend pivots — all cells at once ───────
    trend_pivots = {}
    if has_time and len(all_dates) > 0:
        for kpi in trend_kpis:
            if kpi not in df_norm.columns:
                continue
            try:
                pivot = df_norm.pivot_table(
                    index='cell_id',
                    columns='_timestamp',
                    values=kpi,
                    aggfunc='mean'
                ).reindex(columns=all_dates)
                trend_pivots[kpi] = pivot
            except:
                pass

    period_counts = df_norm.groupby('cell_id').size().to_dict()

    # ── Process each cell ──────────────────────────────────
    results    = []
    cell_store = {}   # reset store

    for _, latest in latest_df.iterrows():
        cell_id = str(latest['cell_id'])

        m = CellMeasurement(
            cell_id=cell_id,
            sinr_db=float(latest['sinr_db']),
            cqi_mean=float(latest['cqi_mean']),
            bler_dl=float(latest['bler_dl']),
            prb_util_dl=float(latest['prb_util_dl']),
            prb_util_ul=float(latest['prb_util_ul']),
            ho_success_rate=float(latest['ho_success_rate']),
            abnormal_release_ratio=float(latest['abnormal_release_ratio']),
            rlf_count=float(latest.get('rlf_count', 0.0)),
            dl_tput_user=float(latest.get('dl_tput_user', 10.0))
        )
        result         = build_result(m)
        result['rank'] = 0

        # Build trend
        trend = {}
        for kpi in trend_kpis:
            if kpi not in trend_pivots:
                continue
            pivot = trend_pivots[kpi]
            if cell_id not in pivot.index:
                continue
            row      = pivot.loc[cell_id]
            vals     = [round(float(v), 2) if pd.notna(v) else None for v in row.values]
            non_null = [v for v in vals if v is not None]
            if not non_null:
                continue
            first_val = next(v for v in vals if v is not None)
            last_val  = next(v for v in reversed(vals) if v is not None)
            diff      = last_val - first_val
            if kpi in good_high:
                trend_dir = 'degrading' if diff < -0.5 else 'improving' if diff > 0.5 else 'stable'
            else:
                trend_dir = 'degrading' if diff > 0.5 else 'improving' if diff < -0.5 else 'stable'
            trend[kpi] = {
                'values':  vals,
                'dates':   date_labels,
                'min':     round(min(non_null), 2),
                'max':     round(max(non_null), 2),
                'mean':    round(sum(non_null)/len(non_null), 2),
                'trend':   trend_dir,
                'periods': len(non_null),
            }

        result['trend']        = trend
        result['period_count'] = period_counts.get(cell_id, 1)

        # Store full result with trend in memory
        cell_store[cell_id] = result

        # Lightweight version for bulk response — no trend
        lightweight = {k: v for k, v in result.items() if k != 'trend'}
        results.append(lightweight)

    results.sort(key=lambda x: x['health_score'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1
        # Also update rank in cell_store
        if r['cell_id'] in cell_store:
            cell_store[r['cell_id']]['rank'] = r['rank']

    bands = [r['health_band'] for r in results]
    return {
        "total_cells":       len(results),
        "total_rows":        len(df_norm),
        "periods":           unique_periods,
        "date_range":        f"{date_labels[0]} to {date_labels[-1]}" if date_labels else "N/A",
        "filtered_inactive": filtered,
        "critical":          bands.count('critical'),
        "degraded":          bands.count('degraded'),
        "watch":             bands.count('watch'),
        "healthy":           bands.count('healthy'),
        "perfect":           bands.count('perfect'),
        "anomalous":         sum(1 for r in results if r['anomaly_label'] == 'anomalous'),
        "worst_cell":        results[0]['cell_id']  if results else None,
        "best_cell":         results[-1]['cell_id'] if results else None,
        "column_mapping":    match_log,
        "cells":             results   # no trend data — lightweight
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
