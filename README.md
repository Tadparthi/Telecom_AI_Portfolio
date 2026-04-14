# Telecom AI Portfolio

I spent 20 years optimizing 5G and 4G networks for T-Mobile, Verizon,
AT&T, Bell Canada, Nokia, and Mavenir. This is what happens when you
combine that with machine learning.

Three projects. Real data. Deployed tools. Built to solve problems I
actually dealt with in the field.

---

## Screenshots

![Dashboard](screenshots/dashboard.png)
*2,038 unique cells ranked by health score — upload any Nokia/Ericsson/Samsung OSS export*

![Cell Detail](screenshots/cell_detail.png)
*Click any cell — KPI vs threshold table + 14-day sparkline trend charts*

![AI Assistant](screenshots/ai_assistant.png)
*GPT-4o powered assistant — ask questions about your network in plain English*

---

## Run it yourself

**Requirements:** Python 3.10+, pip

```bash
# 1. Clone
git clone https://github.com/Tadparthi/Telecom_AI_Portfolio
cd Telecom_AI_Portfolio

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
# Create a file called .env in the project folder:
# OPENAI_API_KEY=sk-your-key-here
# (get a key at platform.openai.com — free credits available)

# 5. Start the tool
# Windows: double-click Launch_NOC_Tool.bat
# Mac/Linux:
uvicorn network_health_api:app --host 0.0.0.0 --port 8000

# 6. Open dashboard
# Go to http://localhost:8000/dashboard in your browser
```

**No Nokia data? No problem.**
A demo CSV file is included — `demo_data.csv`. Upload it to see the
tool in action with anonymized real network data from a live 5G network.

---

## What it does

### KPI Upload Analyzer — bulk network health assessment

Drop a CSV or Excel export from your OSS. The tool figures out which
columns are which — it knows that `NR PRB util PDSCH (NR_5114a)` means
PRB utilization downlink, and `NR Average UE related SINR for PUSCH in
Rank 1 (NR_5062b)` means SINR. You don't have to rename anything.

Multi-period files get grouped by unique cell automatically. A 14-day
Nokia export with 26,790 rows becomes a ranked list of 2,038 cells,
scored on the most recent period.

Click any row and you get two things: a KPI table showing exactly which
thresholds were breached and by how much, and a 14-day sparkline for
each KPI so you can see whether things are getting better or worse.

The trend detection is what makes this useful in practice. A cell with
abnormal release at 18% is worth investigating. A cell where abnormal
release has gone from 11% to 18% over two weeks is urgent.

---

### AI Network Assistant

After uploading a KPI file click **AI Assistant** in the top right.
Ask questions in plain English — the assistant has full context of
your network data and fetches 14-day trend history on demand.

```
You:  "What are my worst cells and what's causing the issues?"

AI:   "Your top critical cell is Cell_0319. SINR has been
       stable at 3.2-4.7dB for 14 days — this isn't a sudden
       failure, it's a chronic coverage problem. More urgent:
       abnormal release has been climbing from 11.7% to 18.6%
       over the same period. Two KPIs degrading together like
       this points to hardware — connector or antenna — not
       interference. Interference would show SINR variability
       and affect neighboring cells similarly.
       
       Recommend field dispatch before HO success rate drops.
       Want me to pull the full trend for any specific cell?"
```

```
You:  "What parameter changes do you recommend?"

AI:   "For Cell_1706 — CQI and HO success issues:
       • A3 offset: set to 3dB
       • TTT: increase to 240ms (suburban profile)
       • RSRQ threshold: verify -10 to -14dB range
       
       For Cell_0226 — SINR and HO issues:
       • Antenna tilt/azimuth optimization first
       • A3 offset: 3dB after antenna work
       • RSRQ threshold check"
```

**Security:** OpenAI calls are proxied through the FastAPI backend.
Your API key lives in a local `.env` file — never in the HTML or JS.

---

### Single Cell Tool

For investigating individual cells. Enter KPI values manually or
load presets. Four presets included: critical, healthy, watch, anomalous.

---

### REST API

The ML models are served as a REST API:

```bash
# Single cell prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sinr_db": 2.1, "cqi_mean": 5.2, "bler_dl": 14.5,
    "prb_util_dl": 85.0, "ho_success_rate": 91.0,
    "abnormal_release_ratio": 18.5, "rlf_count": 67000,
    "prb_util_ul": 72.0, "cell_id": "Cell_1042"
  }'
```

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

Full API docs at `http://localhost:8000/docs` after starting the server.

---

## The ML projects

### Project 1 — Network Cell Health Monitor
`kpi_anomaly.ipynb` · Real 5G NR OSS data · 1,915 cells · 14 days

Three-layer detection system:

| Layer | Technique | Result |
|-------|-----------|--------|
| Rule-based | Weighted KPI health score | 5 health bands |
| ML | Isolation Forest anomaly detection | 1,341 anomalous cell-days |
| Statistical | Linear regression + p-value | 140 degrading cells |

Cell 1123 flagged 5 days before connector failure confirmed —
HO success rate collapsed to 0% on the final two days.
378 cells caught by anomaly detection that passed every threshold check.

---

### Project 2 — Handover Failure Root Cause Classifier
`project2_ho_classifier.ipynb`

Two Random Forest classifiers — A3 intra-frequency and A5
inter-frequency — built with physically correct trigger logic.

| Model | Accuracy | CV Score |
|-------|----------|----------|
| A3 — 5 classes | 99.7% | 0.998 ± 0.001 |
| A5 — 4 classes | 99.3% | 0.995 ± 0.001 |

Failure classes: coverage gap · TTT mismatch · congestion block ·
wrong target · success. Each class maps to a specific NOC action.

---

### Project 3 — LTE Coverage Predictor
`project3_coverage_predictor.ipynb` · 567,195 measurements · Vienna

Random Forest regression predicting RSRP at unmeasured locations.
Features engineered from propagation physics — Haversine distance,
angle off boresight, log10(distance), interference ratio.

Data leakage identified and corrected: initial model used pathloss_db
(R²=0.9683) — found to be mathematically circular with RSRP. Corrected
model: RMSE 4.50 dBm, R²=0.8239 — competitive with commercial tools.

Output: interactive OpenStreetMap heatmap with 4 toggleable layers.

---

## Why domain knowledge matters here

The RLF counter accumulates randomly rather than resetting daily —
using the raw value is meaningless. You need to know that.

RSRQ not SINR drives handover decisions — building a HO classifier
with SINR as the key feature would be wrong. You need to know that.

Angle off boresight looks unimportant in a bar chart because cell
selection masks the effect. The Random Forest found it was the most
important feature (0.213). You need field experience to know whether
to trust the chart or the model.

These aren't things you learn from a Kaggle dataset.

---

## Stack

```
Python · FastAPI · Uvicorn · Pydantic · python-dotenv
Pandas · NumPy · Scikit-learn · SciPy · Folium
Matplotlib · Seaborn · OpenAI API
```

---

## Files

```
kpi_anomaly.ipynb                  Project 1 — Network health monitor
project2_ho_classifier.ipynb       Project 2 — HO failure classifier
project3_coverage_predictor.ipynb  Project 3 — Coverage predictor

network_health_api.py              FastAPI backend + AI proxy
kpi_upload_dashboard.html          Bulk KPI upload analyzer + AI chat
noc_dashboard.html                 Single cell NOC tool
Launch_NOC_Tool.bat                One-click Windows launcher
requirements.txt                   Dependencies

demo_data.csv                      Anonymized 5G network data for testing

screenshots/
  dashboard.png                    Summary cards + ranked cell table
  cell_detail.png                  KPI detail popup + trend charts
  ai_assistant.png                 AI assistant in action
```

---

## Contact

20 years RF · T-Mobile · Verizon · AT&T · Bell Canada · Nokia · Mavenir

Transitioning to AI/ML engineering for telecom networks.
Open to: AI-RAN · Network Operations AI · Telecom AI Engineering

*2026*
