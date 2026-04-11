# Telecom AI Portfolio
### RF Engineer → AI Engineer | 20 Years 5G/4G Network Optimization

---

## Who This Is For

If you are hiring for a role at the intersection of **telecom domain expertise
and machine learning** — network operations AI, AI-RAN, O-RAN intelligence,
or 5G optimization — this portfolio was built for you.

Every model, every threshold, every feature, and every failure class in these
projects comes from 20 years of hands-on RF engineering experience optimizing
4G/5G networks. This is not a generic ML portfolio applied to telecom data.
This is telecom expertise expressed as production-grade ML systems.

---

## Projects

### Project 1 — Network Cell Health Monitor
`kpi_anomaly.ipynb`

**The problem:** A network operations team managing 1,915 cells cannot
manually investigate every cell every day. Problems are found reactively —
after users complain or alarms fire. By then it is too late.

**What was built:** An automated pipeline that analyzes every cell daily,
identifies anomalous behavior invisible to threshold-based monitoring,
detects cells degrading over time, and delivers a prioritized action list
every morning before the NOC team arrives.

**Data:** Real 5G NR OSS counter export — 26,810 cell-days × 650 counters
× 1,915 cells × 14 days

**Three-layer architecture:**

| Layer | Technique | Output |
|-------|-----------|--------|
| Rule-based | Weighted KPI health score | 5 health bands — perfect to critical |
| ML | Isolation Forest anomaly detection | 1,341 anomalous cell-days |
| Statistical | Linear regression trend detection | 140 degrading cells |

**Key results:**
- 378 anomalous cells caught that threshold monitoring completely missed
- Cell 1123 identified as highest risk — connector failure confirmed
  from 14-day trace, HO success rate collapsed to 0% on final two days
- System flagged the degradation **5 days before complete failure**
- 140 cells with statistically significant declining trends identified

**ML modules:** `IsolationForest`, `StandardScaler`, `scipy.stats.linregress`,
`pandas`, `matplotlib`, `seaborn`

**Why Isolation Forest:** Unsupervised — no labels needed. Finds unusual
multivariate KPI combinations invisible to individual thresholds. A cell
with SINR 6dB, CQI 7, HO success 97%, abnormal release 14% triggers no
single alarm — but the combination is anomalous. Isolation Forest catches it.

---

### Project 2 — Handover Failure Root Cause Classifier
`project2_ho_classifier.ipynb`

**The problem:** When handover success rate drops, an RF engineer spends
30-60 minutes pulling counters, checking RSRP, RSRQ, UE speed, TTT, and
target PRB to diagnose the root cause. Across thousands of daily HO failures
this is not scalable.

**What was built:** Two separate classifiers — one for A3 intra-frequency
handover events, one for A5 inter-frequency handover events — that
automatically diagnose root cause and generate specific recommended actions
in milliseconds.

**Why two models:**
A3 and A5 use fundamentally different trigger logic:
- A3: `target RSRP > serving RSRP + offset` (relative measurement)
- A5: `serving RSRP < threshold1 AND target RSRP > threshold2` (absolute)

Combining them would mix relative and absolute measurement logic —
physically incorrect. Two separate models, each physically accurate.

**Network parameters embedded in training data:**
```
A3 offset:           3 dB
RSRQ threshold:     -12 dB
PRB rejection:       60%
TTT values:         160ms (highway) / 240ms (suburban) / 320ms (pedestrian)
A5 threshold 1:    -110 dBm
A5 threshold 2:    -100 dBm
Scenario:           5G NR to 5G NR inter-frequency
```

**Failure classes:**

| Class | A3 | A5 | Description |
|-------|----|----|-------------|
| success | ✓ | ✓ | HO completed normally |
| coverage_gap | ✓ | ✓ | Target cell too weak |
| too_late | ✓ | ✓ | TTT mismatch with UE speed |
| wrong_target | ✓ | — | Pilot pollution, RSRQ filter not enabled |
| congestion_block | ✓ | ✓ | Target PRB > 60%, admission rejected |

**Results:**

| Model | Classes | Accuracy | CV Score | RMSE gap |
|-------|---------|----------|----------|----------|
| A3 classifier | 5 | 99.7% | 0.998 ± 0.001 | 0.3% |
| A5 classifier | 4 | 99.3% | 0.995 ± 0.001 | 0.5% |

**Daily report output:**
```
Date: 2021-07-26 | Total failures: 700

Coverage gap:     168 (33.6%) [HIGH]   → check antenna tilt, feeder/VSWR
Too late HO:      120 (24.0%) [MEDIUM] → reduce TTT to 160ms
Congestion block: 105 (21.0%) [HIGH]   → load balancing, capacity expansion
Wrong target:      60 (12.0%) [MEDIUM] → enable RSRQ filter, check neighbors

Confidence: 81.8% high confidence → immediate action
            14.2% medium → verify first
             4.0% low    → manual review
```

**ML modules:** `RandomForestClassifier`, `train_test_split`,
`cross_val_score`, `confusion_matrix`, `classification_report`,
`predict_proba`

---

### Project 3 — LTE Coverage Predictor
`project3_coverage_predictor.ipynb`

**The problem:** Drive tests cover roads and sample routes. A city has
millions of locations. Coverage prediction tools are expensive, require
3D building databases, and need specialist operators. ML can learn
propagation patterns from existing drive test data and predict coverage
everywhere.

**Data:** 567,195 real LTE drive test measurements — Vienna, Austria
- Three operators (A, B, C)
- 903 matched cells with site parameters
- Multiple frequency bands: 800MHz, 1800MHz, 2600MHz
- Measurements: RSRP, RSRQ, SINR, path loss, throughput
- Cell info: latitude, longitude, height, azimuth

**Feature engineering — domain knowledge as code:**

| Feature | How computed | RF significance |
|---------|-------------|-----------------|
| `distance_m` | Haversine formula (UE→site) | Primary path loss driver |
| `log_distance` | log10(distance) | Linearizes path loss model |
| `angle_off_boresight` | \|bearing_to_UE − azimuth\| | Antenna gain pattern |
| `height_m` | From cell info | Coverage range |
| `freq_ghz` | frequency_khz / 1e6 | Propagation exponent |
| `interference_ratio` | RSSI − RSRP | Interference level |

**Critical finding — data leakage identified:**

Initial model achieved R² = 0.9683 with `pathloss_db` as a feature.
Feature importance analysis revealed pathloss contributed 70.9% —
essentially a mathematical tautology since path loss is defined as
a function of RSRP. This is data leakage.

Two models built to demonstrate understanding:

| Model | Features | RMSE | R² | Use case |
|-------|----------|------|-----|----------|
| V1 with pathloss | 11 | 1.91 dBm | 0.9683 | Interpolation |
| V2 without pathloss | 10 | 4.50 dBm | 0.8239 | True prediction |

V2 is the operationally useful model — predicts coverage at locations
where no measurement exists, using only GPS coordinates and cell geometry.
RMSE of 4.50 dBm is competitive with commercial planning tools.

**Honest evaluation caveat:**
Standard random 80/20 train/test split was used. Drive test measurements
are spatially dense — test points are often meters from training points,
creating spatial autocorrelation. True prediction accuracy at genuinely
unmeasured locations (spatial cross-validation) would be 7-12 dBm.
This limitation is acknowledged and would be addressed in production
with geographic zone holdout validation.

**Coverage quality accuracy (V2):**
- Good coverage detection (> -85 dBm): 89.2%
- Poor coverage detection (< -100 dBm): 92.0%

**Output — Interactive coverage map:**
Generated using `folium` on OpenStreetMap base layer:
- Drive test heatmap overlay (50,000 points)
- Predicted coverage grid (2,500 ML predictions)
- 903 clickable cell site markers with antenna parameters
- Poor coverage alert layer — specific locations flagged
- Toggleable layers — compare actual vs predicted

**ML modules:** `RandomForestRegressor`, `mean_squared_error`,
`mean_absolute_error`, `r2_score`, `folium`, `HeatMap`

---

## Skills Demonstrated

**Python data stack:**
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy`

**ML techniques:**
Supervised classification · Regression · Unsupervised anomaly detection ·
Feature engineering · Train/test splitting · Cross-validation ·
Overfitting prevention · Feature importance · Confusion matrix ·
Precision/recall/F1 · Confidence scoring

**Telecom domain:**
5G NR KPI analysis · LTE/NR handover optimization ·
Drive test data processing · Coverage prediction ·
OSS counter interpretation · RLF analysis ·
A3/A5 event classification · RSRP/RSRQ/SINR analysis ·
Path loss modeling · Antenna geometry

**Production thinking:**
Data leakage identification · Spatial autocorrelation awareness ·
Evaluation methodology critique · Operational report generation ·
Recommendation engine design · Interactive visualization

---

## Why This Background Matters for Telecom AI

Most ML engineers understand the algorithms but not the network.
Most RF engineers understand the network but not the algorithms.

The combination — 20 years of 5G domain expertise plus ML implementation
skills — is rare and valuable. The features that make these models work
cannot be engineered without deep network knowledge:

- Knowing that RLF counters accumulate without daily reset
  and require delta computation rather than raw values
- Knowing that RSRQ not SINR drives handover decisions
  requiring separate A3 and A5 models
- Knowing that angle-off-boresight matters for coverage
  but is masked by cell selection in dense urban environments
- Knowing that path loss and RSRP have a tautological relationship
  that creates data leakage if not identified

These insights come from field experience, not from reading papers.

---

## Target Applications

These projects map directly to active hiring areas:

**AI-RAN / O-RAN:**
Project 1 is a network monitoring rApp.
Project 2 is a root cause diagnosis xApp.
Both align with O-RAN Alliance AI/ML framework specifications.

**Network Operations AI:**
Automated anomaly detection replacing threshold monitoring.
Root cause classification replacing manual L3 trace analysis.
Early warning systems preventing reactive maintenance.

**Coverage Planning AI:**
ML-based coverage prediction replacing expensive commercial tools.
Drive test data augmentation for unmeasured locations.
Coverage hole identification for field investigation prioritization.

---

## Stack

```
Python 3.10
pandas · numpy · scipy · matplotlib · seaborn
scikit-learn · folium
```

---

## Contact

20 years 4G/5G RF engineering experience
Transitioning to AI/ML engineering for telecom networks
Open to roles in: AI-RAN · Network Operations AI · Telecom AI Engineering

---

*Built during RF Engineer → AI Engineer transition | 2026*
