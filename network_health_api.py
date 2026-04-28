
import pickle
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import re
import json
import io
import pandas as pd
import httpx
import uvicorn
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ── ChromaDB RAG setup ─────────────────────────────────────
rag_client     = None
rag_collection = None

def init_rag():
    global rag_client, rag_collection
    try:
        chroma_path    = str(Path(__file__).parent / 'chroma_db')
        rag_client     = chromadb.PersistentClient(path=chroma_path)
        rag_collection = rag_client.get_collection("telecom_params")
        print(f"RAG loaded — {rag_collection.count()} parameters indexed")
    except Exception as e:
        print(f"RAG not available: {e}")

init_rag()
# ── Beam health model setup ────────────────────────────────
beam_model_package = None
beam_store         = {}

def init_beam_model():
    global beam_model_package
    try:
        model_path = Path(__file__).parent / 'mimo_beam_model.pkl'
        with open(model_path, 'rb') as f:
            beam_model_package = pickle.load(f)
        print(f"Beam model loaded — scenarios: {beam_model_package['scenarios']}")
    except Exception as e:
        print(f"Beam model not available: {e}")

init_beam_model()
# ── App ────────────────────────────────────────────────────
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

# In-memory cell store
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

# ── Core ML functions ──────────────────────────────────────
def compute_health_score(m: CellMeasurement):
    flags = []
    score = 0.0
    kpi_detail = []
    kpi_values = {
        'sinr_db': m.sinr_db, 'cqi_mean': m.cqi_mean,
        'bler_dl': m.bler_dl, 'rlf_count': m.rlf_count,
        'abnormal_release_ratio': m.abnormal_release_ratio,
        'prb_util_dl': m.prb_util_dl, 'ho_success_rate': m.ho_success_rate,
        'dl_tput_user': m.dl_tput_user or 0, 'prb_util_ul': m.prb_util_ul,
    }
    for kpi, val in kpi_values.items():
        t = THRESHOLDS[kpi]
        flagged = False
        if t['threshold'] is not None:
            if t['direction'] == 'below' and val < t['threshold']:
                flagged = True
                flags.append("RLF spike" if kpi == 'rlf_count' else f"{t['label']} poor")
                score += t['weight']
            elif t['direction'] == 'above' and val > t['threshold']:
                flagged = True
                flags.append("RLF spike" if kpi == 'rlf_count' else f"{t['label']} high")
                score += t['weight']
        threshold_str = (
            f"{'<' if t['direction'] == 'below' else '>'} {t['threshold']}{t['unit']}"
            if t['threshold'] is not None else 'Monitor only'
        )
        kpi_detail.append({
            'kpi': t['label'], 'value': round(val, 2), 'unit': t['unit'],
            'threshold': t['threshold'], 'threshold_str': threshold_str,
            'direction': t['direction'], 'flagged': flagged, 'weight': t['weight']
        })
    return score, flags, kpi_detail

def score_to_band(score):
    if score == 0:       return "perfect",  "#2ecc71", "none",     "No action required — all KPIs within threshold", 0.99
    elif score <= 1.5:   return "healthy",  "#27ae60", "low",      "Monitor — minor KPI flag detected", 0.95
    elif score <= 3.5:   return "watch",    "#f39c12", "medium",   "Investigate — multiple KPI flags, schedule review", 0.90
    elif score <= 6.0:   return "degraded", "#e67e22", "high",     "Urgent — significant degradation, field check recommended", 0.85
    else:                return "critical", "#e74c3c", "critical", "Critical — immediate field investigation required", 0.80

def compute_anomaly(m: CellMeasurement):
    score = 0.0
    reasons = []
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
        action   = "Anomalous KPI pattern detected despite normal thresholds — investigate"
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
# ── Beam KPI column detection ──────────────────────────────
BEAM_KPI_KEYWORDS = {
    'sinr_db':             ['sinr','nr_5062b','sinr for pusch in rank 1',
                            'pusch_sinr','dl_sinr','sinr for pusch'],
    'rsrp_dbm':            ['rsrp','reference signal received power',
                            'nr_5001','ss-rsrp','dl rsrp'],
    'rsrq_db':             ['rsrq','reference signal received quality',
                            'nr_5002','ss-rsrq','dl rsrq'],
    'cqi_mean':            ['cqi','channel quality','nr_5060b','nr_5061b',
                            'wideband cqi','average wideband cqi'],
    'avg_rank':            ['rank','ri','rank indicator','average rank',
                            'nr_5071a','nr_5071b','dl rank','mimo rank',
                            'average dl rank','mean rank'],
    'mcs_dl_mean':         ['mcs','modulation coding','dl mcs',
                            'mean mcs','average mcs','nr_5080'],
    'bler_dl_pct':         ['bler','block error','bler dl','dl bler',
                            'nr_5054a','initial bler in downlink'],
    'ul_bler_pct':         ['ul bler','bler ul','uplink bler',
                            'nr_5055a','initial bler in uplink'],
    'harq_retx_ratio':      ['harq_retx_ratio','harq_retx','harq',
                        'retransmission','harq retx',
                        'nr_5090','harq retransmission ratio'],
    'prb_util_dl_pct':     ['prb dl','dl prb','prb util dl',
                            'nr_5114a','pdsch prb','prb utilization dl'],
    'ul_prb_util_pct':     ['ul_prb_util_pct','ul_prb_util','prb ul',
                        'ul prb','prb util ul','nr_5115a',
                        'pusch prb','prb utilization ul'],
    'active_users':         ['active_users','active user',
                        'connected user','ue count',
                        'scheduled user','nr_5200','active ue'],
    'traffic_volume_gb':   ['traffic_volume_gb','traffic_volume',
                        'traffic','volume','throughput gb',
                        'dl traffic','data volume'],
    'mu_mimo_ratio':       ['mu_mimo_ratio','mu_mimo','mu mimo',
                        'multi user mimo','mu-mimo',
                        'nr_5072','mu mimo ratio'],
    'rank_1_ratio':        ['rank_1_ratio','rank1_ratio','rank 1',
                        'single layer','rank1','nr_5073','su mimo ratio'],
    'bfr_count':           ['beam failure','bfr','beam failure recovery',
                            'nr_5036','beam failure count'],
    'beam_switch_count':      ['beam_switch_count','beam_switch','beam switch',
                        'beam change','ssb switch',
                        'nr_5037','beam switching count'],
    'beam_concentration':  ['beam_concentration','beam concentration',
                            'dominant beam','beam distribution'],
    'rrc_setup_success_pct':['rrc setup','rrc success','rrc establishment',
                             'nr_5010','rrc setup success ratio'],
    'drb_setup_success_pct':['drb setup','bearer setup','drb success',
                             'nr_5011','data radio bearer'],
    'prach_success_pct':   ['prach','random access','prach success',
                            'nr_5034','prach success rate'],
    'pucch_util_pct':      ['pucch','uplink control','pucch util',
                            'nr_5116','pucch utilization'],
    'ul_rssi_dbm':          ['ul_rssi_dbm','ul_rssi','ul rssi',
                        'uplink rssi','rssi ul','interference',
                        'ul interference','nr_5120',
                        'received interference power'],
    'cell_id':             ['cell','gnodeb','gnb','cell_id','cellname',
                            'cell name','nrcell','object'],
    'date':                ['date','period','time','timestamp'],
}

def detect_beam_columns(df):
    """Auto-detect beam KPI columns from Nokia/Ericsson/Samsung export."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapped     = {}
    match_log  = []
    for kpi, keywords in BEAM_KPI_KEYWORDS.items():
        for keyword in keywords:
            for col_lower, col_orig in cols_lower.items():
                if keyword in col_lower:
                    mapped[kpi]  = col_orig
                    match_log.append(f"{kpi} → '{col_orig}'")
                    break
            if kpi in mapped:
                break
    required  = ['sinr_db','cqi_mean','avg_rank','bler_dl_pct','mu_mimo_ratio']
    unmatched = [k for k in required if k not in mapped]
    return mapped, match_log, unmatched
def auto_detect_columns(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    mapped = {}
    match_log = []
    for kpi, keywords in KPI_KEYWORDS.items():
        found = False
        for keyword in keywords:
            for col_lower, col_orig in cols_lower.items():
                if keyword in col_lower:
                    mapped[kpi] = col_orig
                    match_log.append(f"{kpi} → '{col_orig}'")
                    found = True
                    break
            if found:
                break
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
                except:
                    pass
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    return col, parsed
            except:
                pass
    return None, None

# ── RAG hybrid search helper ───────────────────────────────
def hybrid_score(meta, doc, query, query_embedding=None):
    """Compute hybrid relevance score for a RAG result."""
    full_name  = str(meta.get('full_name',   '') or '').lower()
    abbrev     = str(meta.get('abbrev_name', '') or '').lower()
    mo_raw     = meta.get('mo_class', '')
    mo_class   = str(mo_raw).strip().upper() if mo_raw and str(mo_raw) != 'nan' else ''
    rec_val    = str(meta.get('rec_value', '') or '').lower()
    query_lower = query.lower()

    stop_words = {'for','the','and','with','of','in','to','a','an'}
    keywords   = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stop_words]

    kw = 0.0
    for k in keywords:
        if k in full_name:    kw += 0.15
        if k in abbrev:       kw += 0.10
        if k in doc.lower():  kw += 0.03

    # Event type matching (a1-a5, b1-b2)
    q_events = set(re.findall(r'\ba[1-5]\b|\bb[1-2]\b', query_lower))
    r_events = set(re.findall(r'\ba[1-5]\b|\bb[1-2]\b', full_name + ' ' + abbrev))
    if q_events and r_events:
        if q_events.isdisjoint(r_events): kw -= 0.4
        elif q_events == r_events:        kw += 0.3

    # RSRP vs RSRQ
    if 'rsrp' in full_name and 'rsrq' not in query_lower: kw += 0.05
    if 'rsrq' in full_name and 'rsrq' not in query_lower: kw -= 0.03

    # MO class boost
    inter_freq = any(k in query_lower for k in ['inter','nrhoif','b1','b2','a5','inter-frequency','nr ho interface'])
    if mo_class in ('NRCELL', 'NR CELL') and not inter_freq:       kw += 0.08
    elif mo_class in ('NRHOIF', 'NR HO INTERFACE') and not inter_freq: kw -= 0.05

    # Recommended value quality
    if rec_val and rec_val not in ('n/a','na','','see exception table','nan'): kw += 0.05

    return kw

# ── Agent tools ────────────────────────────────────────────
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_network_summary",
            "description": "Get a high level summary of the entire network — total cells, counts per health band, anomaly count.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_worst_cells",
            "description": "Get the worst performing cells in the network ranked by health score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of worst cells to return. Default 10.", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cell_detail",
            "description": "Get full KPI detail and 14-day trend for a specific cell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_id": {"type": "string", "description": "The cell ID to investigate e.g. Cell_0319"}
                },
                "required": ["cell_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_parameters",
            "description": "Search the 5G NR parameter knowledge base for recommended values and ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Parameter to search e.g. 'A3 offset RSRP'"},
                    "limit": {"type": "integer", "description": "Number of results. Default 3.", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]

def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "get_network_summary":
        if not cell_store:
            return "No network data available. Please upload a KPI file first."
        cells = list(cell_store.values())
        bands = [c['health_band'] for c in cells]
        return json.dumps({
            "total_cells": len(cells),
            "critical":    bands.count('critical'),
            "degraded":    bands.count('degraded'),
            "watch":       bands.count('watch'),
            "healthy":     bands.count('healthy'),
            "perfect":     bands.count('perfect'),
            "anomalous":   sum(1 for c in cells if c['anomaly_label'] == 'anomalous'),
        })

    elif tool_name == "get_worst_cells":
        limit = tool_args.get("limit", 10)
        if not cell_store:
            return "No network data available. Please upload a KPI file first."
        worst = sorted(cell_store.values(), key=lambda x: x['health_score'], reverse=True)[:limit]
        return json.dumps([{
            "cell_id":      c['cell_id'],
            "health_band":  c['health_band'],
            "health_score": c['health_score'],
            "flags":        c['flags'],
            "priority":     c['priority'],
            "anomaly":      c['anomaly_label'],
            "period_count": c.get('period_count', 1),
        } for c in worst])

    elif tool_name == "get_cell_detail":
        cell_id = tool_args.get("cell_id")
        if not cell_id:
            return "cell_id is required"
        if cell_id not in cell_store:
            return f"Cell {cell_id} not found. Upload a KPI file first."
        c = cell_store[cell_id]
        result = {
            "cell_id": c['cell_id'], "health_band": c['health_band'],
            "health_score": c['health_score'], "flags": c['flags'],
            "priority": c['priority'], "anomaly_label": c['anomaly_label'],
            "anomaly_reasons": c.get('anomaly_reasons', []),
            "recommended_action": c['recommended_action'],
            "kpi_detail": c.get('kpi_detail', []),
            "trend_summary": {}
        }
        if c.get('trend'):
            for kpi, t in c['trend'].items():
                if t:
                    result['trend_summary'][kpi] = {
                        "min": t.get('min'), "max": t.get('max'),
                        "mean": t.get('mean'), "trend": t.get('trend'),
                        "periods": t.get('periods'),
                    }
        return json.dumps(result)

    elif tool_name == "search_parameters":
        query = tool_args.get("query", "")
        limit = tool_args.get("limit", 3)
        if not query:
            return "query is required"
        if rag_collection is None:
            return "Parameter knowledge base not available"
        try:
            import openai as oai
            oai_client = oai.OpenAI(api_key=OPENAI_API_KEY)
            embed_response = oai_client.embeddings.create(
                model="text-embedding-3-small", input=query
            )
            query_embedding = embed_response.data[0].embedding
            results = rag_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 4, rag_collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            hits = []
            for i in range(len(results['ids'][0])):
                meta     = results['metadatas'][0][i]
                doc      = results['documents'][0][i]
                distance = results['distances'][0][i]
                sem      = 1 - distance
                kw       = hybrid_score(meta, doc, query)
                hits.append({'score': sem + kw, 'meta': meta})
            hits.sort(key=lambda x: x['score'], reverse=True)
            hits = hits[:limit]
            text = f"Parameter search results for '{query}':\n\n"
            for i, h in enumerate(hits):
                m = h['meta']
                text += f"{i+1}. {m.get('full_name','')} ({m.get('abbrev_name','')})\n"
                text += f"   MO Class: {m.get('mo_class','')}\n"
                text += f"   Recommended: {m.get('rec_value','N/A')}\n"
                text += f"   Range: {m.get('range','')}\n\n"
            return text
        except Exception as e:
            return f"Parameter search error: {str(e)}"

    return f"Unknown tool: {tool_name}"

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Network Cell Health Monitor API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__), "kpi_upload_dashboard.html"))

@app.get("/single")
def serve_single():
    return FileResponse(os.path.join(os.path.dirname(__file__), "noc_dashboard.html"))

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
        "anomalous": sum(1 for r in results if r['anomaly_label'] == 'anomalous'),
        "predictions": results
    }

@app.get("/cell/{cell_id}")
def get_cell_detail_endpoint(cell_id: str):
    if cell_id in cell_store:
        return cell_store[cell_id]
    return {"error": f"Cell {cell_id} not found. Upload a file first."}

@app.post("/chat")
async def chat_proxy(request: dict):
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured on server"}
    messages   = request.get('messages', [])
    max_tokens = request.get('max_tokens', 800)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'gpt-4o', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0.3}
            )
            return response.json()
    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/agent")
async def run_agent(request: dict):
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured"}
    goal      = request.get("goal", "Investigate the network and report the top issues")
    max_steps = request.get("max_steps", 10)
    system_prompt = """You are an expert 5G/LTE network operations AI agent with 20 years of RF engineering experience.
Investigate network issues autonomously and produce a structured report.
Use tools strategically:
1. get_network_summary — understand overall picture
2. get_worst_cells — identify priority cells
3. get_cell_detail — investigate critical cells
4. search_parameters — look up real parameter values before making recommendations
Produce a structured report with: Executive Summary, Critical Findings, Pattern Analysis, Root Cause Assessment, Prioritized Actions.
Be specific — use actual cell IDs, KPI values, and parameter values from the knowledge base."""
    messages    = [{"role": "system", "content": system_prompt}, {"role": "user", "content": goal}]
    steps_taken = []
    step_count  = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        while step_count < max_steps:
            step_count += 1
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'gpt-4o', 'messages': messages, 'tools': AGENT_TOOLS, 'tool_choice': 'auto', 'max_tokens': 2000}
            )
            data    = response.json()
            message = data['choices'][0]['message']
            messages.append(message)
            if message.get('tool_calls'):
                for tool_call in message['tool_calls']:
                    tool_name   = tool_call['function']['name']
                    tool_args   = json.loads(tool_call['function']['arguments'])
                    tool_result = execute_tool(tool_name, tool_args)
                    steps_taken.append({
                        "step": step_count, "tool": tool_name, "args": tool_args,
                        "result": tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    })
                    messages.append({"role": "tool", "tool_call_id": tool_call['id'], "content": tool_result})
            else:
                return {
                    "goal": goal, "steps_taken": len(steps_taken),
                    "tool_calls": steps_taken, "report": message.get('content', 'No report generated')
                }
    return {
        "goal": goal, "steps_taken": len(steps_taken),
        "tool_calls": steps_taken, "report": "Max steps reached",
        "messages": messages[-1].get('content', '')
    }

@app.get("/rag/search")
async def rag_search(query: str, limit: int = 5):
    if rag_collection is None:
        return {"error": "RAG not initialized. Run build_rag.py first."}
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            embed_response = await client.post(
                'https://api.openai.com/v1/embeddings',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'},
                json={'model': 'text-embedding-3-small', 'input': query}
            )
            query_embedding = embed_response.json()['data'][0]['embedding']

        results = rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit * 4, rag_collection.count()),
            include=['documents', 'metadatas', 'distances']
        )

        hits = []
        for i in range(len(results['ids'][0])):
            try:
                meta           = results['metadatas'][0][i]
                doc            = str(results['documents'][0][i] or '')
                distance       = results['distances'][0][i]
                semantic_score = 1 - distance
                kw             = hybrid_score(meta, doc, query)
                combined       = semantic_score + kw
                hits.append({
                    'parameter':      str(meta.get('full_name',   '') or ''),
                    'abbrev':         str(meta.get('abbrev_name', '') or ''),
                    'mo_class':       str(meta.get('mo_class',    '') or ''),
                    'rec_value':      str(meta.get('rec_value',   '') or 'N/A'),
                    'range':          str(meta.get('range',        '') or ''),
                    'category':       str(meta.get('category',     '') or ''),
                    'semantic_score': round(semantic_score, 3),
                    'keyword_score':  round(kw,             3),
                    'relevance':      round(combined,        3),
                    'full_text':      doc,
                })
            except Exception as row_err:
                print(f"RAG row {i} error: {row_err}")
                continue

        hits.sort(key=lambda x: x['relevance'], reverse=True)
        hits = hits[:limit]
        for i, h in enumerate(hits):
            h['rank'] = i + 1

        return {"query": query, "method": "hybrid — semantic + keyword boost", "results": hits}

    except Exception as e:
        return {"error": str(e)}

@app.post("/rag/ask")
async def rag_ask(request: dict):
    if rag_collection is None:
        return {"error": "RAG not initialized"}
    question = request.get("question", "")
    if not question:
        return {"error": "question is required"}
    search_result = await rag_search(question, limit=5)
    if "error" in search_result:
        return search_result
    context = "\n\n".join([
        f"Parameter {r['rank']}: {r['parameter']}\n{r['full_text']}"
        for r in search_result['results']
    ])
    system_prompt = """You are an expert 5G NR network optimization engineer with 20 years of field experience.
Answer questions about 5G parameters using ONLY the parameter documentation provided in the context.
Rules:
- Always cite the exact parameter name and abbreviated name
- If a recommended value exists state it clearly
- If the value says 'see exception table' explain it is scenario-dependent and varies by deployment type
- If the value is N/A explain it is interdependent on other parameters
- Always mention the valid range
- Be specific and actionable"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ],
                    "max_tokens": 600, "temperature": 0.1
                }
            )
            data   = response.json()
            answer = data["choices"][0]["message"]["content"]
        return {
            "question": question, "answer": answer,
            "parameters_used": [r["parameter"] for r in search_result["results"]],
            "sources": search_result["results"]
        }
    except Exception as e:
        return {"error": str(e)}
# ── Beam health endpoints ──────────────────────────────────
@app.post("/beam/upload")
async def beam_upload(file: UploadFile = File(...)):
    """
    Upload a Nokia/Ericsson/Samsung 5G NR beam KPI export.
    Auto-detects column names, computes derived features,
    runs three-tier detection (IF → RF → confidence gate).
    """
    global beam_store

    if beam_model_package is None:
        return {"error": "Beam model not loaded. Run project4_mimo_analyzer.ipynb first to generate mimo_beam_model.pkl"}

    contents = await file.read()
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

    # ── Detect columns ─────────────────────────────────────
    mapped, match_log, unmatched = detect_beam_columns(df)

    if unmatched:
        return {
            "error":        f"Could not find required beam KPI columns: {unmatched}",
            "your_columns": list(df.columns)[:30],
            "matched":      match_log,
            "hint":         "File needs rank, CQI, MU-MIMO ratio, BLER, and SINR columns"
        }

    # ── Build normalized dataframe ─────────────────────────
    df_norm = pd.DataFrame()
    for kpi, col in mapped.items():
        if kpi not in ['cell_id','date']:
            df_norm[kpi] = pd.to_numeric(df[col], errors='coerce')

    df_norm['cell_id'] = (df[mapped['cell_id']].astype(str)
                          if 'cell_id' in mapped
                          else [f"Cell_{i:04d}" for i in range(len(df))])

    if 'date' in mapped:
        for fmt in ['%m/%d/%Y','%Y-%m-%d','%d/%m/%Y']:
            try:
                df_norm['date'] = pd.to_datetime(df[mapped['date']],
                                                  format=fmt, errors='coerce')
                if df_norm['date'].notna().sum() > len(df)*0.5:
                    break
            except:
                pass
    else:
        df_norm['date'] = pd.NaT

    # Fill missing optional KPIs with neutral defaults
    optional_defaults = {
        'rsrp_dbm':             -85.0,
        'rsrq_db':              -10.0,
        'mcs_dl_mean':           20.0,
        'ul_bler_pct':            3.0,
        'harq_retx_ratio':        0.1,
        'ul_prb_util_pct':       30.0,
        'active_users':          10.0,
        'traffic_volume_gb':      3.0,
        'rank_1_ratio':           0.4,
        'bfr_count':              2.0,
        'beam_switch_count':      5.0,
        'beam_concentration':     0.3,
        'rrc_setup_success_pct': 98.0,
        'drb_setup_success_pct': 98.0,
        'prach_success_pct':     97.0,
        'pucch_util_pct':        25.0,
        'ul_rssi_dbm':         -110.0,
        'prb_util_dl_pct':       40.0,
        'load_index':             0.7,
    }
    for col, default in optional_defaults.items():
        if col not in df_norm.columns:
            df_norm[col] = default

    df_norm = df_norm.dropna(subset=['sinr_db','cqi_mean',
                                      'avg_rank','bler_dl_pct']).reset_index(drop=True)

    if len(df_norm) == 0:
        return {"error": "No valid rows after cleaning"}

    # ── Latest period per cell ─────────────────────────────
    has_time = df_norm['date'].notna().any()
    if has_time:
        latest = df_norm.sort_values('date').groupby('cell_id').last().reset_index()
    else:
        latest = df_norm.groupby('cell_id').last().reset_index()

    # ── Compute derived features ───────────────────────────
    latest['cqi_rank_gap'] = (
        latest['cqi_mean'] / np.maximum(latest['avg_rank'], 0.5)
    ).round(3)
    latest['beam_efficiency'] = np.clip(
        (latest['avg_rank'] / 4.0) *
        (1 - latest['bler_dl_pct'] / 100) *
        latest['mu_mimo_ratio'],
        0, 1
    ).round(3)
    latest['sinr_cqi_delta'] = (
        latest['sinr_db'] - latest['cqi_mean'] * 1.5
    ).round(2)

    # ── Load model components ──────────────────────────────
    rf              = beam_model_package['rf']
    iso             = beam_model_package['iso']
    scaler          = beam_model_package['scaler']
    le              = beam_model_package['le']
    FEATURE_COLS    = beam_model_package['feature_cols']
    CONF_THRESHOLD  = 0.60  # overrride — 70% too strict for step-change scenarios

    # Fill any still-missing features with column mean or 0
    for col in FEATURE_COLS:
        if col not in latest.columns:
            latest[col] = 0.0

    X = latest[FEATURE_COLS].fillna(0)

    # ── Three-tier detection ───────────────────────────────
    X_scaled       = scaler.transform(X)
    iso_scores     = iso.decision_function(X_scaled)
    iso_preds      = iso.predict(X_scaled)
    rf_probas      = rf.predict_proba(X)
    rf_preds       = rf.predict(X)

    results  = []
    beam_store = {}

    PRIORITY_MAP = {
        'beam_failure':      'critical',
        'beam_misalignment': 'high',
        'su_mimo_fallback':  'medium',
        'healthy':           'none',
        'unknown_anomaly':   'high',
    }

    SCENARIO_COLORS = {
        'healthy':           '#3fb950',
        'su_mimo_fallback':  '#f0883e',
        'beam_misalignment': '#d29922',
        'beam_failure':      '#f85149',
        'unknown_anomaly':   '#bc8cff',
    }

    for i, row in latest.iterrows():
        cell_id      = str(row['cell_id'])
        iso_score    = float(iso_scores[i])
        is_anomalous = iso_preds[i] == -1
        proba        = rf_probas[i]
        confidence   = float(np.max(proba))
        pred_label   = le.inverse_transform([rf_preds[i]])[0]
        all_proba    = dict(zip(le.classes_,
                           [round(float(p),3) for p in proba]))

        # Determine final label
        if not is_anomalous:
            final_label = 'healthy'
        elif confidence < CONF_THRESHOLD:
            final_label = 'unknown_anomaly'
        else:
            final_label = pred_label

        # Rule-based flags
        flags = []
        recs  = []
        cqr   = float(row.get('cqi_rank_gap', 0))
        rank  = float(row.get('avg_rank', 2))
        bfc   = float(row.get('bfr_count', 0))
        bsc   = float(row.get('beam_switch_count', 0))
        bcon  = float(row.get('beam_concentration', 0))
        mu    = float(row.get('mu_mimo_ratio', 0.5))
        prs   = float(row.get('prach_success_pct', 98))
        harq  = float(row.get('harq_retx_ratio', 0.1))
        users = float(row.get('active_users', 5))
        bler  = float(row.get('bler_dl_pct', 3))
        mcs   = float(row.get('mcs_dl_mean', 20))
        be    = float(row.get('beam_efficiency', 0.4))

        if cqr > 7.0 and rank < 1.8:
            flags.append(f"SU-MIMO fallback — CQI/rank gap {cqr:.1f} (normal <5.0)")
            recs.append("Drive test to check UE spatial distribution")
            recs.append("Review SSB beam configuration and sector split")
        if bcon > 0.65:
            flags.append(f"High beam concentration {bcon:.2f} — UEs angularly clustered")
            recs.append("Consider cell azimuth adjustment or sector split")
        if mu < 0.20 and users > 8:
            flags.append(f"Low MU-MIMO ratio {mu:.2f} with {int(users)} active users")
            recs.append("Verify MU-MIMO pairing is enabled in scheduler")
        if bsc > 15:
            flags.append(f"High beam switching {int(bsc)}/day — UEs searching for stable beam")
            recs.append("Audit SSB beam sweep configuration")
            recs.append("Check for antenna mechanical issues (tilt/azimuth drift)")
        if bfc > 20:
            flags.append(f"High BFR count {int(bfc)}/day — beam failures occurring")
            recs.append("Immediate field investigation — possible hardware fault")
        if prs < 93:
            flags.append(f"Low PRACH success {prs:.1f}% — initial access beam issue")
            recs.append("Review SSB beam sweep pattern for coverage gaps")
        if harq > 0.20:
            flags.append(f"High HARQ retransmission ratio {harq:.2f} — post-beam quality poor")
        if bler > 12:
            flags.append(f"High DL BLER {bler:.1f}% — link quality degraded")
        if be < 0.10:
            flags.append(f"Low beam efficiency score {be:.2f} — spatial multiplexing underutilized")
        if final_label == 'unknown_anomaly':
            flags.append("Unknown anomaly — does not match any trained failure pattern")
            recs.append("Manual investigation required")
            recs.append(f"Closest trained match: {pred_label} ({confidence:.0%}) — verify")

        # KPI detail for popup
        kpi_detail = [
            {'kpi': 'SINR',              'value': round(float(row.get('sinr_db',0)),2),   'unit': 'dB',  'good_above': 12},
            {'kpi': 'RSRP',              'value': round(float(row.get('rsrp_dbm',0)),2),  'unit': 'dBm', 'good_above': -85},
            {'kpi': 'RSRQ',              'value': round(float(row.get('rsrq_db',0)),2),   'unit': 'dB',  'good_above': -11},
            {'kpi': 'CQI Mean',          'value': round(float(row.get('cqi_mean',0)),2),  'unit': '',    'good_above': 9},
            {'kpi': 'Avg Rank',          'value': round(float(row.get('avg_rank',0)),2),  'unit': '',    'good_above': 2.0},
            {'kpi': 'MCS DL Mean',       'value': round(float(row.get('mcs_dl_mean',0)),2),'unit':'',   'good_above': 18},
            {'kpi': 'BLER DL',           'value': round(float(row.get('bler_dl_pct',0)),2),'unit': '%', 'good_above': None, 'good_below': 10},
            {'kpi': 'UL BLER',           'value': round(float(row.get('ul_bler_pct',0)),2),'unit': '%', 'good_above': None, 'good_below': 10},
            {'kpi': 'HARQ Retx Ratio',   'value': round(float(row.get('harq_retx_ratio',0)),3),'unit':'','good_above': None,'good_below': 0.15},
            {'kpi': 'PRB Util DL',       'value': round(float(row.get('prb_util_dl_pct',0)),1),'unit':'%','good_above': None,'good_below': 80},
            {'kpi': 'Active Users',      'value': int(row.get('active_users',0)),           'unit': '',   'good_above': None},
            {'kpi': 'MU-MIMO Ratio',     'value': round(float(row.get('mu_mimo_ratio',0)),3),'unit':'',  'good_above': 0.45},
            {'kpi': 'Rank 1 Ratio',      'value': round(float(row.get('rank_1_ratio',0)),3),'unit': '',  'good_above': None,'good_below': 0.55},
            {'kpi': 'BFR Count',         'value': int(row.get('bfr_count',0)),              'unit': '/day','good_above': None,'good_below': 15},
            {'kpi': 'Beam Switch Count', 'value': int(row.get('beam_switch_count',0)),      'unit': '/day','good_above': None,'good_below': 12},
            {'kpi': 'Beam Concentration','value': round(float(row.get('beam_concentration',0)),3),'unit':'','good_above':None,'good_below':0.55},
            {'kpi': 'PRACH Success',     'value': round(float(row.get('prach_success_pct',0)),1),'unit':'%','good_above': 95},
            {'kpi': 'RRC Setup Success', 'value': round(float(row.get('rrc_setup_success_pct',0)),1),'unit':'%','good_above': 96},
            {'kpi': 'UL RSSI',           'value': round(float(row.get('ul_rssi_dbm',0)),1), 'unit': 'dBm','good_above': None,'good_below': -100},
            {'kpi': 'CQI/Rank Gap',      'value': round(float(row.get('cqi_rank_gap',0)),2),'unit': '',  'good_above': None,'good_below': 7.0},
            {'kpi': 'Beam Efficiency',   'value': round(float(row.get('beam_efficiency',0)),3),'unit':'', 'good_above': 0.30},
            {'kpi': 'SINR-CQI Delta',    'value': round(float(row.get('sinr_cqi_delta',0)),2),'unit':'dB','good_above':None,'good_below':3.0},
        ]

        # Flag each KPI
        for k in kpi_detail:
            v = k['value']
            if k.get('good_above') is not None:
                k['flagged']   = bool(v < k['good_above'])
                k['threshold'] = f">{k['good_above']}"
            elif k.get('good_below') is not None:
                k['flagged']   = bool(v > k['good_below'])
                k['threshold'] = f"<{k['good_below']}"
            else:
                k['flagged']   = False
                k['threshold'] = 'info'

        result = {
            'cell_id':         cell_id,
            'final_label':     final_label,
            'priority':        PRIORITY_MAP.get(final_label, 'high'),
            'color':           SCENARIO_COLORS.get(final_label, '#666666'),
            'iso_score':       round(iso_score, 4),
            'is_anomalous':    bool(is_anomalous),
            'rf_scenario':     pred_label,
            'rf_confidence':   round(confidence, 3),
            'all_proba':       all_proba,
            'flags':           flags,
            'recommendations': list(dict.fromkeys(recs)),
            'flag_count':      len(flags),
            'kpi_detail':      kpi_detail,
            'cqi_rank_gap':    round(cqr, 2),
            'avg_rank':        round(rank, 2),
            'beam_efficiency': round(be, 3),
            'mu_mimo_ratio':   round(mu, 3),
        }
        results.append(result)
        beam_store[cell_id] = result

    # Sort by priority
    priority_order = {'critical':0,'high':1,'medium':2,'none':3}
    results.sort(key=lambda x: priority_order.get(x['priority'],4))
    for i, r in enumerate(results):
        r['rank'] = i + 1

    labels     = [r['final_label'] for r in results]
    priorities = [r['priority']    for r in results]

    return {
        'total_cells':      len(results),
        'healthy':          labels.count('healthy'),
        'su_mimo_fallback': labels.count('su_mimo_fallback'),
        'beam_misalignment':labels.count('beam_misalignment'),
        'beam_failure':     labels.count('beam_failure'),
        'unknown_anomaly':  labels.count('unknown_anomaly'),
        'critical':         priorities.count('critical'),
        'high':             priorities.count('high'),
        'medium':           priorities.count('medium'),
        'column_mapping':   match_log,
        'cells':            results,
    }


@app.get("/beam/cell/{cell_id}")
def get_beam_cell(cell_id: str):
    if cell_id in beam_store:
        return beam_store[cell_id]
    return {"error": f"Cell {cell_id} not found. Upload a beam KPI file first."}
@app.post("/predict/upload")
async def predict_from_file(file: UploadFile = File(...)):
    global cell_store
    contents = await file.read()
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

    time_col, time_series = detect_time_column(df)
    mapped, match_log, unmatched = auto_detect_columns(df)
    required_strict = ['sinr_db','cqi_mean','bler_dl','prb_util_dl','prb_util_ul','ho_success_rate','abnormal_release_ratio']
    if unmatched:
        strict_missing = [k for k in unmatched if k in required_strict]
        if strict_missing:
            return {"error": f"Could not find columns for: {strict_missing}", "your_columns": list(df.columns), "matched_so_far": match_log}

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
    df_norm['_timestamp'] = time_series.values if time_series is not None else pd.NaT

    df_norm  = df_norm.dropna(subset=required_strict).reset_index(drop=True)
    before   = len(df_norm)
    df_norm  = df_norm[df_norm['cqi_mean'] > 0].reset_index(drop=True)
    filtered = before - len(df_norm)
    if len(df_norm) == 0:
        return {"error": "No valid rows after cleaning."}

    has_time       = df_norm['_timestamp'].notna().any()
    unique_periods = int(df_norm['_timestamp'].nunique()) if has_time else 1
    all_dates      = sorted(df_norm['_timestamp'].dropna().unique()) if has_time else []
    date_labels    = [pd.Timestamp(d).strftime('%m/%d') for d in all_dates]
    trend_kpis     = ['sinr_db','cqi_mean','bler_dl','prb_util_dl','prb_util_ul','ho_success_rate','abnormal_release_ratio','dl_tput_user']
    good_high      = {'sinr_db','cqi_mean','ho_success_rate','dl_tput_user'}

    if has_time:
        latest_df = df_norm.sort_values('_timestamp').groupby('cell_id').last().reset_index()
    else:
        latest_df = df_norm.groupby('cell_id').last().reset_index()

    trend_pivots = {}
    if has_time and len(all_dates) > 0:
        for kpi in trend_kpis:
            if kpi not in df_norm.columns:
                continue
            try:
                pivot = df_norm.pivot_table(index='cell_id', columns='_timestamp', values=kpi, aggfunc='mean').reindex(columns=all_dates)
                trend_pivots[kpi] = pivot
            except:
                pass

    period_counts = df_norm.groupby('cell_id').size().to_dict()
    results       = []
    cell_store    = {}

    for _, latest in latest_df.iterrows():
        cell_id = str(latest['cell_id'])
        m = CellMeasurement(
            cell_id=cell_id,
            sinr_db=float(latest['sinr_db']), cqi_mean=float(latest['cqi_mean']),
            bler_dl=float(latest['bler_dl']), prb_util_dl=float(latest['prb_util_dl']),
            prb_util_ul=float(latest['prb_util_ul']), ho_success_rate=float(latest['ho_success_rate']),
            abnormal_release_ratio=float(latest['abnormal_release_ratio']),
            rlf_count=float(latest.get('rlf_count', 0.0)),
            dl_tput_user=float(latest.get('dl_tput_user', 10.0))
        )
        result         = build_result(m)
        result['rank'] = 0
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
                'values': vals, 'dates': date_labels,
                'min': round(min(non_null), 2), 'max': round(max(non_null), 2),
                'mean': round(sum(non_null)/len(non_null), 2),
                'trend': trend_dir, 'periods': len(non_null),
            }
        result['trend']        = trend
        result['period_count'] = period_counts.get(cell_id, 1)
        cell_store[cell_id]    = result
        lightweight = {k: v for k, v in result.items() if k != 'trend'}
        results.append(lightweight)

    results.sort(key=lambda x: x['health_score'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1
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
        "cells":             results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
