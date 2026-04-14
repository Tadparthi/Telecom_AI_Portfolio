# Telecom AI Portfolio
### RF Engineer → AI Engineer | 20 Years 5G/4G | Real Network Data | Deployed Tools

---

## What this is

A production-grade AI system for 5G/LTE network operations — built by a
senior RF engineer using 20 years of field experience and real network data.

Not a research project. Not a tutorial. A working tool that a NOC team
could use today.

---

## See it in action

```
1. Clone the repo
2. pip install -r requirements.txt
3. Double-click Launch_NOC_Tool.bat
4. Upload any Nokia/Ericsson/Samsung OSS KPI export
```

The tool analyzes every cell, ranks them worst to best, and shows you
exactly why each cell is flagged — with 14 days of trend history.

---

## What it does

### Upload any OSS export → instant ranked health report

Drop a CSV or Excel KPI export from any vendor. The tool auto-detects
column names — Nokia, Ericsson, Samsung — no reformatting needed.

```
NR PRB util PDSCH (NR_5114a)           → detected as PRB Util DL
NR Average UE related SINR for PUSCH   → detected as SINR
NR Intra gNB intra frequency HO ratio  → detected as HO Success Rate
```

Multi-period files are automatically grouped by unique cell. A 14-day
export with 26,790 rows becomes a ranked list of 2,038 unique cells —
health score per cell based on the latest period.

**Output — ranked dashboard:**
```
Rank  Cell ID     Health    Score  Flags  Priority
#1    Cell_1042   CRITICAL  14.5   7      critical
#2    Cell_0887   CRITICAL  12.0   6      critical
#3    Cell_1156   DEGRADED   7.5   4      high
...
#2038 Cell_0034   PERFECT    0.0   0      none
```

Filter by health band, flag count, anomaly status, or search by cell ID.
Export the full ranked list as CSV.

---

### Click any cell → KPI breakdown + 14-day trend

Every cell row is clickable. The popup shows:

**KPI analysis — current period vs thresholds:**
```
KPI               Value      Threshold   Status
SINR              2.1 dB     < 5 dB      ❌ Flagged
CQI Mean          5.2        < 6         ❌ Flagged
BLER DL           8.2%       > 10%       ✓ OK
HO Success Rate   91.2%      < 95%       ❌ Flagged
PRB Util DL       45%        > 80%       ✓ OK
Abnormal Release  12.1%      > 15%       ✓ OK
```

**14-day sparkline charts — one per KPI:**
- Trend direction: stable / improving / degrading
- Min, avg, max across all periods
- Color coded — green improving, red degrading

**Anomaly detection:**
Flags unusual KPI combinations even when individual thresholds are fine.
Example: good SINR but poor CQI — possible interference pattern.

**Recommended action:**
```
Priority:  CRITICAL
Action:    Immediate field investigation required
```

---

### Single cell tool — manual KPI entry

For ad-hoc investigation. Enter KPI values manually or load presets.
Four presets included: critical, healthy, watch, anomalous.

---

### REST API — callable from any system

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sinr_db": 2.1, "cqi_mean": 5.2, "bler_dl": 14.5,
       "prb_util_dl": 85.0, "ho_success_rate": 91.0,
       "abnormal_release_ratio": 18.5, "rlf_count": 67000,
       "prb_util_ul": 72.0, "cell_id": "Cell_1042"}'
```

Response:
```json
{
  "cell_id": "Cell_1042",
  "health_band": "critical",
  "health_score": 14.5,
  "flags": ["SINR poor", "CQI poor", "HO success low"],
  "recommended_action": "Critical — immediate field investigation required",
  "priority": "critical",
  "confidence": 0.80
}
```

Endpoints: `/predict` · `/predict/batch` · `/predict/upload` · `/cell/{id}`

---

## The ML behind it

Three independent ML projects power the system — all built on real
network data with domain-specific feature engineering.

### Project 1 — Network Cell Health Monitor
`kpi_anomaly.ipynb`

**Data:** Real 5G NR OSS export — 26,810 cell-days × 650 KPI counters

Three-layer detection system:

| Layer | Technique | Result |
|-------|-----------|--------|
| Rule-based | Weighted KPI health score | 5 health bands |
| ML | Isolation Forest anomaly detection | 1,341 anomalous cell-days |
| Statistical | Linear regression + p-value | 140 degrading cells |

Key result: Cell 1123 flagged 5 days before connector failure —
HO success rate collapsed to 0% confirmed on final two days.

378 anomalous cells caught that threshold monitoring completely missed.

---

### Project 2 — Handover Failure Root Cause Classifier
`project2_ho_classifier.ipynb`

Two separate Random Forest classifiers — A3 intra-frequency (5 classes)
and A5 inter-frequency (4 classes) — built with physically correct
trigger logic from 20 years of field experience.

| Model | Accuracy | CV Score | Classes |
|-------|----------|----------|---------|
| A3 classifier | 99.7% | 0.998 ± 0.001 | coverage_gap · too_late · wrong_target · congestion · success |
| A5 classifier | 99.3% | 0.995 ± 0.001 | coverage_gap · too_late · congestion · success |

Recommendation engine maps predictions to NOC actions:
```
coverage_gap     → check antenna tilt and feeder/VSWR
too_late HO      → reduce TTT to 160ms
congestion_block → load balancing or capacity expansion
wrong_target     → enable RSRQ filter at -12dB
```

---

### Project 3 — LTE Coverage Predictor + Interactive Map
`project3_coverage_predictor.ipynb`

**Data:** 567,195 real LTE drive test measurements — Vienna, 3 operators, 903 cells

Random Forest regression model predicting RSRP at unmeasured locations
using propagation geometry features engineered from domain knowledge:

```python
distance_m          = haversine(UE_position, site_position)
log_distance        = log10(distance_m)          # linearizes path loss
angle_off_boresight = |bearing_to_UE - azimuth|  # antenna gain pattern
interference_ratio  = RSSI - RSRP                # interference level
```

**Data leakage identified and corrected:** Initial model achieved R²=0.9683
with pathloss_db — found to be mathematically tautological with RSRP (70.9%
feature importance). Corrected model without pathloss: RMSE 4.50 dBm,
R²=0.8239 — competitive with commercial planning tools.

Output: Interactive OpenStreetMap coverage heatmap with 4 toggleable layers.

---

## Why domain knowledge is the differentiator

These models work because of what went into the features — not because
of the algorithms. Examples:

```
A3 and A5 use different trigger logic
→ separate models required — one combined model is physically wrong

RLF counters accumulate randomly not daily
→ raw values unusable — binary spike flag required

Pathloss and RSRP are mathematically related
→ using pathloss as a feature is data leakage — caught and corrected

Angle-off-boresight appears unimportant in bar charts
→ Random Forest found it most important feature (0.213) — hidden by
  cell selection masking off-boresight UEs in aggregate statistics
```

A pure ML engineer cannot make these design decisions without years of
RF network experience. A pure RF engineer cannot build the pipeline.
This portfolio demonstrates both.

---

## Stack

```
Python · FastAPI · Uvicorn · Pydantic
Pandas · NumPy · Scikit-learn · SciPy · Folium
Matplotlib · Seaborn
```

---

## Repository structure

```
Telecom_AI_Portfolio/
│
├── kpi_anomaly.ipynb                  Project 1 — Network health monitor
├── project2_ho_classifier.ipynb       Project 2 — HO failure classifier
├── project3_coverage_predictor.ipynb  Project 3 — Coverage predictor
│
├── network_health_api.py              FastAPI backend
├── kpi_upload_dashboard.html          Bulk KPI upload analyzer
├── noc_dashboard.html                 Single cell NOC tool
├── launch_noc.py                      Python launcher
├── Launch_NOC_Tool.bat                One-click Windows launcher
├── requirements.txt                   Dependencies
│
├── network_health_dashboard.png       Project 1 dashboard output
├── ho_confusion_matrices.png          Project 2 confusion matrices
├── coverage_heatmap.png               Project 3 static map
└── coverage_heatmap.html              Project 3 interactive map
```

---

## Contact

20 years 4G/5G RF engineering — T-Mobile · Verizon · AT&T · Bell Canada · Nokia · Mavenir

Transitioning to AI/ML engineering for telecom networks.
Open to roles in: AI-RAN · Network Operations AI · Telecom AI Engineering

*Built during RF Engineer → AI Engineer transition | 2026*
