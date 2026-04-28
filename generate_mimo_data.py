"""
5G Massive MIMO Beam Health Synthetic Data Generator
=====================================================
Generates 500 cells x 30 days of realistic 5G NR beam KPI data.
22 KPIs + 3 derived features covering the full RF/RAN stack.

Scenarios:
  healthy           (40%) - All KPIs nominal, MU-MIMO working
  su_mimo_fallback  (20%) - High CQI/SINR but low rank (UE clustering)
  beam_misalignment (20%) - Poor CQI despite adequate SINR
  beam_failure      (20%) - Hardware/environment degradation with onset day

Key diagnostic signature per scenario:
  su_mimo_fallback:  high cqi_rank_gap + high active_users + high beam_concentration
  beam_misalignment: low MCS vs CQI + high HARQ + high beam_switch_count
  beam_failure:      BFR/beam_switch spike + PRACH drop at failure onset
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ── Config ─────────────────────────────────────────────────
N_CELLS    = 500
N_DAYS     = 30
START_DATE = datetime(2024, 1, 1)
OUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mimo_beam_data.csv')

SCENARIO_DIST = {
    'healthy':           0.40,
    'su_mimo_fallback':  0.20,
    'beam_misalignment': 0.20,
    'beam_failure':      0.20,
}

# ── Noise / pattern helpers ─────────────────────────────────
def daily_load(n_days):
    """Realistic daily load with weekday/weekend variation."""
    dow = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.82, 0.78] * 5)[:n_days]
    noise = np.random.normal(0, 0.03, n_days)
    return np.clip(dow + noise, 0.3, 1.0)

def noise(arr, frac=0.06, lo=None, hi=None):
    result = arr + np.random.normal(0, np.abs(arr) * frac)
    if lo is not None: result = np.clip(result, lo, hi)
    return result

def spikes(arr, prob=0.04, mag=1.5, up=True):
    mask = np.random.random(len(arr)) < prob
    delta = mag * np.abs(arr) * mask
    return arr + delta if up else arr - delta

def trend(n, start, end, std=0.02):
    return np.linspace(start, end, n) + np.random.normal(0, std, n)

def pct(arr): return np.clip(arr, 0, 100)
def ratio(arr): return np.clip(arr, 0, 1)
def pos(arr): return np.maximum(arr, 0)

# ── Scenario generators ─────────────────────────────────────

def gen_healthy(n, load):
    """
    All KPIs nominal. MU-MIMO pairing efficient.
    UEs spatially distributed — scheduler finds good pairs.
    """
    sinr_db               = noise(np.full(n, np.random.uniform(13, 22)), 0.07, 3, 32)
    rsrp_dbm              = noise(np.full(n, np.random.uniform(-85, -65)), 0.04, -120, -44)
    rsrq_db               = noise(np.full(n, np.random.uniform(-10, -6)),  0.05, -20, -3)
    cqi_mean              = noise(np.full(n, np.random.uniform(10, 13)),   0.06, 4, 15)
    avg_rank              = noise(np.full(n, np.random.uniform(2.0, 3.5)), 0.07, 1, 4)
    mcs_dl_mean           = noise(np.full(n, np.random.uniform(22, 27)),   0.05, 0, 28)
    bler_dl_pct           = noise(np.full(n, np.random.uniform(1.5, 4.0)), 0.15, 0.1, 10)
    ul_bler_pct           = noise(np.full(n, np.random.uniform(1.0, 3.5)), 0.15, 0.1, 10)
    harq_retx_ratio       = noise(np.full(n, np.random.uniform(0.05, 0.12)), 0.15, 0, 1)
    prb_util_dl_pct       = pct(noise(load * np.random.uniform(30, 60), 0.10))
    ul_prb_util_pct       = pct(noise(load * np.random.uniform(20, 45), 0.10))
    active_users          = pos(noise(load * np.random.uniform(8, 20), 0.15)).astype(int)
    traffic_volume_gb     = pos(noise(load * np.random.uniform(2, 8), 0.15))
    mu_mimo_ratio         = ratio(noise(np.full(n, np.random.uniform(0.55, 0.80)), 0.08))
    rank_1_ratio          = ratio(1 - mu_mimo_ratio + np.random.normal(0, 0.02, n))
    bfr_count             = np.random.poisson(2, n)
    beam_switch_count     = np.random.poisson(4, n)
    beam_concentration    = noise(np.full(n, np.random.uniform(0.15, 0.35)), 0.12, 0, 1)
    rrc_setup_success_pct = pct(noise(np.full(n, np.random.uniform(98, 99.8)), 0.01, 90, 100))
    drb_setup_success_pct = pct(noise(np.full(n, np.random.uniform(98, 99.8)), 0.01, 90, 100))
    prach_success_pct     = pct(noise(np.full(n, np.random.uniform(97, 99.5)), 0.01, 85, 100))
    pucch_util_pct        = pct(noise(load * np.random.uniform(15, 35), 0.12))
    ul_rssi_dbm           = noise(np.full(n, np.random.uniform(-115, -105)), 0.03, -130, -80)

    return locals()

def gen_su_mimo_fallback(n, load):
    """
    UE angular clustering — all UEs in same azimuth sector.
    Scheduler cannot find spatially separated UE pairs for MU-MIMO.
    Signature: High SINR + High CQI + Low rank + High beam_concentration
               + High active_users (plenty of UEs but all clustered)
    """
    sinr_db               = noise(np.full(n, np.random.uniform(13, 21)), 0.07, 3, 32)
    rsrp_dbm              = noise(np.full(n, np.random.uniform(-83, -63)), 0.04, -120, -44)
    rsrq_db               = noise(np.full(n, np.random.uniform(-10, -6)),  0.05, -20, -3)
    cqi_mean              = noise(np.full(n, np.random.uniform(10, 13)),   0.06, 4, 15)

    # Low rank — the problem — gradual worsening as UE distribution clusters
    rank_base             = np.random.uniform(1.1, 1.6)
    avg_rank              = noise(trend(n, rank_base, rank_base * 0.82, 0.04), 0.06, 1, 4)

    # MCS still good — channel quality is fine, just can't multiplex
    mcs_dl_mean           = noise(np.full(n, np.random.uniform(20, 26)),   0.06, 0, 28)
    bler_dl_pct           = noise(np.full(n, np.random.uniform(2, 5)),     0.15, 0.1, 12)
    ul_bler_pct           = noise(np.full(n, np.random.uniform(1.5, 4)),   0.15, 0.1, 12)
    harq_retx_ratio       = noise(np.full(n, np.random.uniform(0.06, 0.14)), 0.12, 0, 1)
    prb_util_dl_pct       = pct(noise(load * np.random.uniform(40, 70), 0.10))
    ul_prb_util_pct       = pct(noise(load * np.random.uniform(25, 50), 0.10))

    # Many active users — but they are spatially clustered
    active_users          = pos(noise(load * np.random.uniform(12, 25), 0.15)).astype(int)
    traffic_volume_gb     = pos(noise(load * np.random.uniform(3, 9), 0.15))

    # Low MU-MIMO ratio — scheduler falling back to SU-MIMO
    mu_mimo_ratio         = ratio(noise(np.full(n, np.random.uniform(0.08, 0.28)), 0.10))
    rank_1_ratio          = ratio(1 - mu_mimo_ratio + np.random.normal(0, 0.03, n))

    # Few BFRs — beams are fine, just clustered
    bfr_count             = np.random.poisson(2, n)
    beam_switch_count     = np.random.poisson(3, n)

    # HIGH beam concentration — all UEs on same 1-2 beams
    beam_concentration    = noise(np.full(n, np.random.uniform(0.65, 0.90)), 0.08, 0, 1)

    rrc_setup_success_pct = pct(noise(np.full(n, np.random.uniform(97, 99.5)), 0.01, 90, 100))
    drb_setup_success_pct = pct(noise(np.full(n, np.random.uniform(97, 99.5)), 0.01, 90, 100))
    prach_success_pct     = pct(noise(np.full(n, np.random.uniform(96, 99.2)), 0.01, 85, 100))
    pucch_util_pct        = pct(noise(load * np.random.uniform(18, 38), 0.12))
    ul_rssi_dbm           = noise(np.full(n, np.random.uniform(-113, -103)), 0.03, -130, -80)

    return locals()

def gen_beam_misalignment(n, load):
    """
    SSB beam misconfiguration or antenna downtilt error.
    Beams not aligned with UE distribution.
    Signature: Low CQI despite adequate SINR + Low MCS + High HARQ
               + High beam_switch (UEs keep searching for better beam)
               + Low PRACH (initial access beam also misaligned)
    """
    sinr_db               = noise(np.full(n, np.random.uniform(7, 14)), 0.10, 2, 25)
    rsrp_dbm              = noise(np.full(n, np.random.uniform(-100, -80)), 0.05, -120, -44)
    rsrq_db               = noise(np.full(n, np.random.uniform(-14, -9)),   0.07, -20, -3)

    # Low CQI — beam not tracking UEs well
    cqi_mean              = noise(np.full(n, np.random.uniform(5, 9)),    0.10, 3, 15)

    # Unstable rank — beam tracking intermittent
    avg_rank              = noise(np.full(n, np.random.uniform(1.3, 2.2)), 0.14, 1, 4)
    avg_rank              = spikes(avg_rank, prob=0.08, mag=0.5, up=False)

    # Low MCS — scheduler using lower order modulation due to poor CQI
    mcs_dl_mean           = noise(np.full(n, np.random.uniform(12, 18)),  0.10, 0, 28)

    # High BLER and HARQ — post-beamforming quality poor
    bler_dl_pct           = noise(np.full(n, np.random.uniform(8, 18)),   0.15, 0.5, 35)
    bler_dl_pct           = spikes(bler_dl_pct, prob=0.10, mag=1.5)
    ul_bler_pct           = noise(np.full(n, np.random.uniform(6, 14)),   0.15, 0.5, 30)
    harq_retx_ratio       = noise(np.full(n, np.random.uniform(0.18, 0.35)), 0.12, 0, 1)

    prb_util_dl_pct       = pct(noise(load * np.random.uniform(25, 55), 0.10))
    ul_prb_util_pct       = pct(noise(load * np.random.uniform(18, 40), 0.10))
    active_users          = pos(noise(load * np.random.uniform(5, 15), 0.15)).astype(int)
    traffic_volume_gb     = pos(noise(load * np.random.uniform(1, 4), 0.15))
    mu_mimo_ratio         = ratio(noise(np.full(n, np.random.uniform(0.20, 0.45)), 0.12))
    rank_1_ratio          = ratio(1 - mu_mimo_ratio + np.random.normal(0, 0.04, n))

    # High BFR and beam switches — UEs struggling to find good beam
    bfr_count             = np.random.poisson(10, n)
    beam_switch_count     = np.random.poisson(18, n)   # HIGH — UEs constantly re-selecting

    # Moderate concentration — beams exist but wrong direction
    beam_concentration    = noise(np.full(n, np.random.uniform(0.35, 0.60)), 0.10, 0, 1)

    # Low PRACH — initial access beam also misaligned
    rrc_setup_success_pct = pct(noise(np.full(n, np.random.uniform(90, 97)), 0.02, 75, 100))
    drb_setup_success_pct = pct(noise(np.full(n, np.random.uniform(89, 96)), 0.02, 75, 100))
    prach_success_pct     = pct(noise(np.full(n, np.random.uniform(85, 94)), 0.02, 70, 100))
    pucch_util_pct        = pct(noise(load * np.random.uniform(20, 40), 0.12))

    # Higher UL RSSI — misaligned UL beam = more interference from neighbors
    ul_rssi_dbm           = noise(np.full(n, np.random.uniform(-108, -98)), 0.04, -130, -80)

    return locals()

def gen_beam_failure(n, load):
    """
    Hardware degradation or environmental blockage with a clear onset.
    Pre-failure: reasonably healthy.
    Post-failure: SINR drops, BFR spikes, PRACH fails, rank collapses.
    Simulates: antenna connector, hardware degradation, new building.
    """
    failure_day = int(np.random.randint(5, 21))
    fd = failure_day

    def step(pre, post): return np.where(np.arange(n) < fd, pre, post)

    # Pre/post baselines
    sinr_pre, sinr_post   = np.random.uniform(11, 18), None
    sinr_post             = sinr_pre * np.random.uniform(0.35, 0.65)
    sinr_db               = noise(step(sinr_pre, sinr_post), 0.12, 0, 32)
    sinr_db               = spikes(sinr_db, prob=0.12, mag=0.9, up=False)

    rsrp_pre              = np.random.uniform(-88, -68)
    rsrp_post             = rsrp_pre - np.random.uniform(8, 18)
    rsrp_dbm              = noise(step(rsrp_pre, rsrp_post), 0.04, -120, -44)

    rsrq_pre              = np.random.uniform(-11, -7)
    rsrq_post             = rsrq_pre - np.random.uniform(3, 7)
    rsrq_db               = noise(step(rsrq_pre, rsrq_post), 0.06, -20, -3)

    cqi_pre               = np.random.uniform(9, 12)
    cqi_post              = cqi_pre * np.random.uniform(0.45, 0.70)
    cqi_mean              = noise(step(cqi_pre, cqi_post), 0.10, 3, 15)

    rank_pre              = np.random.uniform(1.8, 3.0)
    rank_post             = np.random.uniform(1.0, 1.4)
    avg_rank              = noise(step(rank_pre, rank_post), 0.12, 1, 4)
    avg_rank              = spikes(avg_rank, prob=0.15, mag=0.6, up=False)

    mcs_pre               = np.random.uniform(20, 26)
    mcs_post              = mcs_pre * np.random.uniform(0.40, 0.65)
    mcs_dl_mean           = noise(step(mcs_pre, mcs_post), 0.08, 0, 28)

    bler_pre              = np.random.uniform(2, 5)
    bler_post             = np.random.uniform(12, 28)
    bler_dl_pct           = noise(step(bler_pre, bler_post), 0.20, 0.1, 40)
    bler_dl_pct           = spikes(bler_dl_pct, prob=0.12, mag=2.0)

    ul_bler_pre           = np.random.uniform(1.5, 4)
    ul_bler_post          = np.random.uniform(8, 20)
    ul_bler_pct           = noise(step(ul_bler_pre, ul_bler_post), 0.18, 0.1, 35)

    harq_pre              = np.random.uniform(0.05, 0.12)
    harq_post             = np.random.uniform(0.25, 0.45)
    harq_retx_ratio       = noise(step(harq_pre, harq_post), 0.15, 0, 1)

    prb_util_dl_pct       = pct(noise(load * np.random.uniform(20, 50), 0.10))
    ul_prb_util_pct       = pct(noise(load * np.random.uniform(15, 38), 0.10))

    users_pre             = load * np.random.uniform(6, 16)
    users_post            = load * np.random.uniform(2, 8)   # users drop post-failure
    active_users          = pos(noise(step(users_pre, users_post), 0.18)).astype(int)
    traf_pre              = load * np.random.uniform(2, 6)
    traf_post             = load * np.random.uniform(0.3, 1.5)
    traffic_volume_gb     = pos(noise(step(traf_pre, traf_post), 0.15))

    mu_pre                = np.random.uniform(0.45, 0.70)
    mu_post               = np.random.uniform(0.05, 0.18)
    mu_mimo_ratio         = ratio(noise(step(mu_pre, mu_post), 0.10))
    rank_1_ratio          = ratio(1 - mu_mimo_ratio + np.random.normal(0, 0.03, n))

    # BFR spikes dramatically at failure onset
    bfr_pre_arr           = np.random.poisson(3, fd).astype(float)
    bfr_post_arr          = np.random.poisson(30, n - fd).astype(float)
    bfr_count             = np.concatenate([bfr_pre_arr, bfr_post_arr])

    bsw_pre_arr           = np.random.poisson(5, fd).astype(float)
    bsw_post_arr          = np.random.poisson(22, n - fd).astype(float)
    beam_switch_count     = np.concatenate([bsw_pre_arr, bsw_post_arr])

    bc_pre                = np.random.uniform(0.18, 0.38)
    bc_post               = np.random.uniform(0.45, 0.75)
    beam_concentration    = ratio(noise(step(bc_pre, bc_post), 0.10))

    # PRACH drops after failure — SSB beams affected
    rrc_pre               = np.random.uniform(97.5, 99.5)
    rrc_post              = np.random.uniform(78, 91)
    rrc_setup_success_pct = pct(noise(step(rrc_pre, rrc_post), 0.015, 60, 100))

    drb_pre               = rrc_pre - np.random.uniform(0, 0.5)
    drb_post              = rrc_post - np.random.uniform(0, 1)
    drb_setup_success_pct = pct(noise(step(drb_pre, drb_post), 0.015, 60, 100))

    prach_pre             = np.random.uniform(96, 99.2)
    prach_post            = np.random.uniform(72, 88)
    prach_success_pct     = pct(noise(step(prach_pre, prach_post), 0.02, 60, 100))

    pucch_util_pct        = pct(noise(load * np.random.uniform(12, 30), 0.12))

    # UL RSSI spikes post-failure (noise floor rises with hardware issue)
    rssi_pre              = np.random.uniform(-116, -106)
    rssi_post             = rssi_pre + np.random.uniform(4, 10)
    ul_rssi_dbm           = noise(step(rssi_pre, rssi_post), 0.03, -130, -80)

    result = locals()
    result['failure_day'] = failure_day
    return result

# ── Derived features ───────────────────────────────────────
def compute_derived(d):
    """
    Three derived diagnostic features:

    cqi_rank_gap:       CQI / rank
                        HIGH = good CQI but low rank = SU-MIMO fallback
                        Both low = beam misalignment or failure

    beam_efficiency:    composite score combining rank, BLER, MU-MIMO
                        0 = worst, 1 = best

    sinr_cqi_delta:     SINR - (CQI * 1.5)
                        Positive = SINR good but CQI not following
                        = beam alignment issue
    """
    cqi   = np.array(d['cqi_mean'])
    rank  = np.maximum(np.array(d['avg_rank']), 0.5)
    bler  = np.array(d['bler_dl_pct'])
    mu    = np.array(d['mu_mimo_ratio'])
    sinr  = np.array(d['sinr_db'])

    cqi_rank_gap        = np.round(cqi / rank, 3)
    beam_efficiency     = np.round(np.clip((rank/4.0) * (1 - bler/100) * mu, 0, 1), 3)
    sinr_cqi_delta      = np.round(sinr - cqi * 1.5, 2)

    return cqi_rank_gap, beam_efficiency, sinr_cqi_delta

# ── Main ───────────────────────────────────────────────────
def generate():
    print(f"Generating {N_CELLS} cells x {N_DAYS} days = {N_CELLS * N_DAYS:,} rows...")
    print(f"KPIs per row: 22 + 3 derived features\n")

    dates     = [(START_DATE + timedelta(days=i)).strftime('%m/%d/%Y')
                 for i in range(N_DAYS)]

    n_h  = int(N_CELLS * SCENARIO_DIST['healthy'])
    n_su = int(N_CELLS * SCENARIO_DIST['su_mimo_fallback'])
    n_bm = int(N_CELLS * SCENARIO_DIST['beam_misalignment'])
    n_bf = N_CELLS - n_h - n_su - n_bm

    scenarios = (['healthy']           * n_h  +
                 ['su_mimo_fallback']  * n_su +
                 ['beam_misalignment'] * n_bm +
                 ['beam_failure']      * n_bf)
    np.random.shuffle(scenarios)

    print(f"Scenario split:")
    print(f"  Healthy:           {n_h}  cells (40%)")
    print(f"  SU-MIMO fallback:  {n_su} cells (20%)")
    print(f"  Beam misalignment: {n_bm} cells (20%)")
    print(f"  Beam failure:      {n_bf} cells (20%)\n")

    KPI_COLS = [
        'sinr_db', 'rsrp_dbm', 'rsrq_db', 'cqi_mean',
        'avg_rank', 'mcs_dl_mean', 'bler_dl_pct', 'ul_bler_pct',
        'harq_retx_ratio', 'prb_util_dl_pct', 'ul_prb_util_pct',
        'active_users', 'traffic_volume_gb', 'mu_mimo_ratio',
        'rank_1_ratio', 'bfr_count', 'beam_switch_count',
        'beam_concentration', 'rrc_setup_success_pct',
        'drb_setup_success_pct', 'prach_success_pct',
        'pucch_util_pct', 'ul_rssi_dbm',
    ]

    rows = []
    for idx in range(N_CELLS):
        cell_id  = f"Cell_{idx:04d}"
        scenario = scenarios[idx]
        load     = daily_load(N_DAYS)
        n        = N_DAYS

        failure_day = -1
        if scenario == 'healthy':
            d = gen_healthy(n, load)
        elif scenario == 'su_mimo_fallback':
            d = gen_su_mimo_fallback(n, load)
        elif scenario == 'beam_misalignment':
            d = gen_beam_misalignment(n, load)
        else:
            d = gen_beam_failure(n, load)
            failure_day = d['failure_day']

        cqi_rank_gap, beam_efficiency, sinr_cqi_delta = compute_derived(d)

        for day in range(N_DAYS):
            row = {
                'cell_id':     cell_id,
                'date':        dates[day],
                'scenario':    scenario,
                'failure_day': failure_day,
                'load_index':  round(float(load[day]), 3),
            }
            for col in KPI_COLS:
                val = d[col]
                if isinstance(val, np.ndarray):
                    row[col] = round(float(val[day]), 3)
                else:
                    row[col] = round(float(val), 3)

            row['cqi_rank_gap']    = round(float(cqi_rank_gap[day]),    3)
            row['beam_efficiency'] = round(float(beam_efficiency[day]), 3)
            row['sinr_cqi_delta']  = round(float(sinr_cqi_delta[day]),  3)
            rows.append(row)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{N_CELLS} cells generated...")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(df):,} rows to: mimo_beam_data.csv")
    print(f"Columns: {len(df.columns)}")

    print("\nMean KPIs by scenario:")
    diag_cols = ['sinr_db','rsrp_dbm','cqi_mean','avg_rank','mcs_dl_mean',
                 'bler_dl_pct','bfr_count','mu_mimo_ratio',
                 'beam_concentration','prach_success_pct',
                 'cqi_rank_gap','beam_efficiency']
    summary = df.groupby('scenario')[diag_cols].mean().round(2)
    print(summary.T.to_string())

    print("\nKey diagnostic signatures:")
    sigs = {
        'healthy':           ('cqi_rank_gap',     '<5.0',  'Rank keeping up with CQI'),
        'su_mimo_fallback':  ('cqi_rank_gap',     '>7.0',  'High CQI + Low rank = clustering'),
        'beam_misalignment': ('beam_switch_count','>15',   'UEs searching for better beam'),
        'beam_failure':      ('bfr_count',        '>20',   'Post-failure BFR spike'),
    }
    for sc, (kpi, thresh, reason) in sigs.items():
        val = df[df['scenario']==sc][kpi].mean()
        print(f"  {sc:20s}: {kpi:22s} = {val:.1f}  ({thresh})  {reason}")

    return df

if __name__ == "__main__":
    df = generate()
    print("\nNext step: run project4_mimo_analyzer.ipynb")
