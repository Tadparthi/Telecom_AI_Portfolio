# Telecom AI Portfolio

I spent 20 years optimizing 5G and 4G networks for T-Mobile, Verizon,
AT&T, Bell Canada, Nokia, and Mavenir. This is what happens when you
combine that with machine learning.

Three projects. Real data. Deployed tools. Built to solve problems I
actually dealt with in the field.

---

## Run it yourself

```bash
git clone https://github.com/Tadparthi/Telecom_AI_Portfolio
pip install -r requirements.txt
# Windows: double-click Launch_NOC_Tool.bat
# Mac/Linux: uvicorn network_health_api:app --reload
```

Then open `http://localhost:8000/dashboard` and upload any Nokia,
Ericsson, or Samsung KPI export. No reformatting needed.

---

## The tools

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

### Single Cell Tool — quick investigation

For when you already know which cell you want to look at. Type in the
KPIs, hit run, get the assessment. Four presets included if you just
want to see what the output looks like.

---

### REST API — for integration

The ML models are served as a REST API so you can call them from
anything — a script, a dashboard, an O-RAN xApp, whatever.

```bash
curl -X POST http://localhost:8000/predict \
  -d '{"sinr_db": 2.1, "cqi_mean": 5.2, "bler_dl": 14.5,
       "prb_util_dl": 85.0, "ho_success_rate": 91.0,
       "abnormal_release_ratio": 18.5, "rlf_count": 67000,
       "prb_util_ul": 72.0, "cell_id": "Cell_1042"}'
```

Returns health band, score, flags, anomaly status, and recommended
action. Under a second for a single cell, a few seconds for a full
network export.

---

## The ML projects

### Project 1 — Network Cell Health Monitor
`kpi_anomaly.ipynb` · Real 5G NR OSS data · 1,915 cells · 14 days

This one started with a question I kept running into: how do you find
the cells that are going to fail before they actually fail?

Three layers working together. A rule-based health score using 8 KPI
flags with weights I calibrated from field experience. Isolation Forest
anomaly detection across 18 KPIs simultaneously — catches the cells
that look fine on every individual metric but whose combination of
readings doesn't make sense. And linear regression trend detection
with p-value filtering so you're not chasing noise.

The headline result: Cell 1123 was flagged as highest combined risk
5 days before its HO success rate hit 0%. The 14-day trace confirmed
a connector failure in progress. Threshold monitoring saw nothing.

378 cells were flagged by the anomaly detector that passed every
individual threshold check. That's the gap this fills.

---

### Project 2 — Handover Failure Root Cause Classifier
`project2_ho_classifier.ipynb`

One thing that used to take 30-45 minutes of manual work: pull the
counters, check RSRP delta, look at RSRQ on the target, check PRB
utilization, pull an L3 trace if needed, conclude it was coverage gap
or TTT mismatch or congestion or pilot pollution.

This does that in milliseconds, across every HO failure in the network.

Two separate models — A3 intra-frequency and A5 inter-frequency —
because they use fundamentally different trigger logic. A3 fires on
relative RSRP delta. A5 fires on absolute thresholds. Combining them
into one model would be physically wrong, and the accuracy would
reflect that.

The parameters came directly from real network configs: 3dB A3 offset,
-12dB RSRQ threshold, 60% PRB rejection, TTT values matched to
highway/suburban/pedestrian UE speed profiles.

| Model | Accuracy | CV | Gap |
|-------|----------|----|-----|
| A3 — 5 classes | 99.7% | 0.998 ± 0.001 | 0.3% |
| A5 — 4 classes | 99.3% | 0.995 ± 0.001 | 0.5% |

---

### Project 3 — LTE Coverage Predictor
`project3_coverage_predictor.ipynb` · 567,195 real measurements · Vienna

Trained on real LTE drive test data — three operators, 903 cells, multiple
frequency bands. The model learns the relationship between where you are
relative to the antenna and what signal you get, then predicts signal
at locations that were never driven.

The interesting part of this project isn't the accuracy — it's what
went wrong first. The initial model hit R²=0.9683, which seemed too
good. Feature importance showed pathloss_db at 70.9%. That's a
tautology — path loss is essentially derived from RSRP, so using it to
predict RSRP is circular. Removed it, rebuilt, got RMSE 4.50 dBm and
R²=0.8239 — which is actually competitive with commercial planning tools
and doesn't cheat.

The geometry features — Haversine distance, angle off boresight,
log10(distance) to linearize the path loss relationship — came from
understanding how antennas work, not from reading an ML textbook.

Output is an interactive OpenStreetMap heatmap with four toggleable
layers: drive test measurements, ML predictions, cell site markers
with antenna parameters, and poor coverage alerts.

---

## A note on why domain knowledge matters here

The RLF counter in the Nokia OSS export accumulates randomly rather
than resetting daily. Using the raw value is meaningless. You need to
know that to build a useful health score.

RSRQ — not SINR — drives handover decisions in 4G/5G. Building a HO
failure classifier using SINR as the key feature would be wrong. You
need to know that to build the right features.

Angle off boresight looks unimportant in a bar chart because cell
selection masks the effect — UEs that are badly off-boresight tend to
switch cells before you measure them. The Random Forest found it was
the most important feature anyway (0.213 importance). You need field
experience to know whether to trust the chart or the model.

These aren't things you learn from a Kaggle dataset.

---

## Stack

Python · FastAPI · Scikit-learn · Pandas · NumPy · SciPy · Folium

---

## Files

```
kpi_anomaly.ipynb                  Project 1
project2_ho_classifier.ipynb       Project 2
project3_coverage_predictor.ipynb  Project 3

network_health_api.py              FastAPI backend
kpi_upload_dashboard.html          Bulk upload tool
noc_dashboard.html                 Single cell tool
Launch_NOC_Tool.bat                One-click launcher

coverage_heatmap.html              Interactive coverage map
requirements.txt
```

---

20 years RF · Now building AI for the same problems · Open to
AI-RAN · Network Operations AI · Telecom AI Engineering roles

*2026*
