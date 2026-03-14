import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
BASE_SIGNUP = datetime(2024, 1, 1)
RANDOM_CUSTOMERS = list(range(1000, 1240))     # 240 background customers
SPECIAL_CUSTOMERS = list(range(2000, 2010))    # 10 hand-tuned decisive customers
ALL_CUSTOMERS = RANDOM_CUSTOMERS + SPECIAL_CUSTOMERS

# ============================================================
# 1) CUSTOMERS
# ============================================================
customers = []

for cid in RANDOM_CUSTOMERS:
    customers.append({
        "customer_id": cid,
        "reliability_score": np.random.randint(300, 850),
        "potential_value": np.random.randint(20000, 200000),
        "account_status": np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10]),
        "signup_date": BASE_SIGNUP + timedelta(days=int(np.random.randint(0, 180)))
    })

# Hand-tuned special customers
special_specs = {
    2000: {"reliability": 760, "status": 0, "signup": datetime(2024, 1, 5)},
    2001: {"reliability": 745, "status": 0, "signup": datetime(2024, 1, 5)},
    2002: {"reliability": 735, "status": 0, "signup": datetime(2024, 1, 6)},
    2003: {"reliability": 725, "status": 0, "signup": datetime(2024, 1, 6)},
    2004: {"reliability": 715, "status": 0, "signup": datetime(2024, 1, 7)},
    2005: {"reliability": 780, "status": 0, "signup": datetime(2024, 1, 7)},
    2006: {"reliability": 790, "status": 0, "signup": datetime(2024, 1, 8)},
    2007: {"reliability": 710, "status": 0, "signup": datetime(2024, 1, 8)},
    2008: {"reliability": 695, "status": 1, "signup": datetime(2024, 1, 9)},  # inactive for final ranking, but can affect graph if needed
    2009: {"reliability": 805, "status": 0, "signup": datetime(2024, 1, 9)},
}

for cid, spec in special_specs.items():
    customers.append({
        "customer_id": cid,
        "reliability_score": spec["reliability"],
        "potential_value": 120000 + (cid - 2000) * 1500,
        "account_status": spec["status"],
        "signup_date": spec["signup"]
    })

customers = pd.DataFrame(customers)

# near-duplicates for first 20 random + first 6 special
dup_ids = RANDOM_CUSTOMERS[:20] + SPECIAL_CUSTOMERS[:6]
dups = customers[customers["customer_id"].isin(dup_ids)].copy()
dups["signup_date"] = dups["signup_date"] + pd.to_timedelta(1, unit="us")
customers_full = pd.concat([customers, dups], ignore_index=True)

# ============================================================
# 2) EVENTS
# ============================================================
event_rows = []
event_id = 1

def add_event(cid, etype, d, amount=None, tax_rate=None):
    global event_id
    event_rows.append({
        "event_id": event_id,
        "customer_id": cid,
        "event_type": etype,
        "amount": amount,
        "tax_rate": tax_rate,
        "event_dt": pd.Timestamp(d)
    })
    event_id += 1

# ----------------------------
# Background random events
# ----------------------------
for cid in RANDOM_CUSTOMERS:
    signup = customers.loc[customers["customer_id"] == cid, "signup_date"].iloc[0]
    n_events = np.random.randint(8, 16)

    dates = []
    for i in range(n_events):
        if i < 2 and np.random.rand() < 0.4:
            dt_ = signup - timedelta(days=int(np.random.randint(1, 20)))
        else:
            dt_ = signup + timedelta(days=int(np.random.randint(0, 340)))
        dates.append(dt_)

    dates = sorted(dates)
    active = True
    for dt_ in dates:
        r = np.random.rand()
        if active:
            if r < 0.68:
                etype = "purchase"
            elif r < 0.82:
                etype = "refund"
            elif r < 0.90:
                etype = "pause"
                active = False
            else:
                etype = "purchase"
        else:
            if r < 0.50:
                etype = "resume"
                active = True
            elif r < 0.75:
                etype = "refund"
            else:
                etype = "pause"

        amount = None
        tax_rate = None
        if etype in ("purchase", "refund"):
            amount = float(np.random.randint(20, 500))
            tax_rate = 0.08 + np.random.random() * 0.05

        add_event(cid, etype, dt_, amount, tax_rate)

# ----------------------------
# Hand-tuned special events
# ----------------------------
# Goal:
# - 2000: benefits if refunds are bucket-matched correctly
# - 2001: benefits only if adjustments replace, not add
# - 2002/2003: sensitive to reciprocal monthly contribution
# - 2004/2005: sensitive to blocked pair removal
# - 2006/2007/2009: close competitors

# Helper comments:
# taxed value bucket = floor((amount*(1+tax))/50)

# 2000: one refund should cancel only a same-bucket purchase, not the last arbitrary one
add_event(2000, "purchase", "2024-02-01", 99.0, 0.10)   # 108.9 -> bucket 2
add_event(2000, "purchase", "2024-02-03", 181.0, 0.10)  # 199.1 -> bucket 3
add_event(2000, "refund",   "2024-02-05", 95.0, 0.10)   # 104.5 -> bucket 2, should cancel bucket-2 purchase
add_event(2000, "purchase", "2024-03-01", 300.0, 0.10)  # 330 -> bucket 6
add_event(2000, "purchase", "2024-03-15", 250.0, 0.10)  # 275 -> bucket 5
add_event(2000, "refund",   "2024-03-20", 247.0, 0.10)  # 271.7 -> bucket 5, cancels only bucket-5 purchase
add_event(2000, "purchase", "2024-04-02", 320.0, 0.10)
add_event(2000, "purchase", "2024-05-10", 340.0, 0.10)

# 2001: adjustment month, wrong add-vs-replace will inflate it
add_event(2001, "purchase", "2024-02-01", 210.0, 0.10)
add_event(2001, "purchase", "2024-02-10", 220.0, 0.10)
add_event(2001, "purchase", "2024-03-01", 260.0, 0.10)
add_event(2001, "purchase", "2024-03-12", 260.0, 0.10)
add_event(2001, "purchase", "2024-04-01", 310.0, 0.10)
add_event(2001, "purchase", "2024-05-01", 330.0, 0.10)

# 2002 and 2003: same-ish totals, graph should separate them via reciprocal monthly eligibility
add_event(2002, "purchase", "2024-02-02", 280.0, 0.10)
add_event(2002, "purchase", "2024-03-02", 300.0, 0.10)
add_event(2002, "purchase", "2024-04-02", 320.0, 0.10)
add_event(2002, "purchase", "2024-05-02", 340.0, 0.10)

add_event(2003, "purchase", "2024-02-02", 285.0, 0.10)
add_event(2003, "purchase", "2024-03-02", 295.0, 0.10)
add_event(2003, "purchase", "2024-04-02", 325.0, 0.10)
add_event(2003, "purchase", "2024-05-02", 335.0, 0.10)

# 2004 and 2005: blocked-pair sensitivity
add_event(2004, "purchase", "2024-02-04", 260.0, 0.10)
add_event(2004, "purchase", "2024-03-04", 260.0, 0.10)
add_event(2004, "purchase", "2024-04-04", 300.0, 0.10)
add_event(2004, "purchase", "2024-05-04", 320.0, 0.10)

add_event(2005, "purchase", "2024-02-04", 250.0, 0.10)
add_event(2005, "purchase", "2024-03-04", 255.0, 0.10)
add_event(2005, "purchase", "2024-04-04", 310.0, 0.10)
add_event(2005, "purchase", "2024-05-04", 330.0, 0.10)

# 2006: slightly stronger own totals, but should be moderated by replacement semantics
add_event(2006, "purchase", "2024-02-06", 360.0, 0.10)
add_event(2006, "purchase", "2024-03-06", 365.0, 0.10)
add_event(2006, "purchase", "2024-04-06", 370.0, 0.10)
add_event(2006, "purchase", "2024-05-06", 375.0, 0.10)

# 2007: middling
add_event(2007, "purchase", "2024-02-06", 270.0, 0.10)
add_event(2007, "purchase", "2024-03-06", 275.0, 0.10)
add_event(2007, "purchase", "2024-04-06", 280.0, 0.10)
add_event(2007, "purchase", "2024-05-06", 285.0, 0.10)

# 2008 inactive final rank, but can contribute
add_event(2008, "purchase", "2024-02-07", 420.0, 0.10)
add_event(2008, "purchase", "2024-03-07", 430.0, 0.10)
add_event(2008, "purchase", "2024-04-07", 440.0, 0.10)

# 2009 strong competitor
add_event(2009, "purchase", "2024-02-08", 350.0, 0.10)
add_event(2009, "purchase", "2024-03-08", 355.0, 0.10)
add_event(2009, "purchase", "2024-04-08", 360.0, 0.10)
add_event(2009, "purchase", "2024-05-08", 365.0, 0.10)

events = pd.DataFrame(event_rows)

# Float trap
money_mask = events["event_type"].isin(["purchase", "refund"])
events.loc[(events.index % 19 == 0) & money_mask, "tax_rate"] = 0.10500000000000001

# NaN amount trap
events.loc[(events.index % 29 == 0) & money_mask, "amount"] = np.nan

# Mixed date formats
def fmt_date(d, i):
    if i % 3 == 0:
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    elif i % 3 == 1:
        return pd.Timestamp(d).strftime("%d/%m/%Y")
    else:
        return pd.Timestamp(d).strftime("%m-%d-%Y")

events = events.sort_values(["customer_id", "event_dt", "event_id"]).reset_index(drop=True)
events["event_date"] = [fmt_date(d, i) for i, d in enumerate(events["event_dt"])]
events = events.drop(columns=["event_dt"])

# ============================================================
# 3) TAG PERIODS
# ============================================================
tag_period_rows = []

def add_tag_period(cid, tag, start_date, end_date):
    tag_period_rows.append({
        "customer_id": cid,
        "tag": tag,
        "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
    })

# random background tag periods
tag_names = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
for cid in RANDOM_CUSTOMERS:
    k = np.random.choice([1, 2, 3], p=[0.45, 0.40, 0.15])
    chosen = np.random.choice(tag_names, size=k, replace=False)
    for tag in chosen:
        n_periods = np.random.choice([1, 2], p=[0.8, 0.2])
        starts = sorted([
            BASE_SIGNUP + timedelta(days=int(np.random.randint(0, 300)))
            for _ in range(n_periods)
        ])
        for s in starts:
            e = s + timedelta(days=int(np.random.randint(40, 120)))
            add_tag_period(cid, tag, s, e)

# hand-tuned special tag periods
# Make month-specific reciprocity matter around Mar/Apr/May 2024
add_tag_period(2000, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2000, "beta",  "2024-03-01", "2024-03-31")

add_tag_period(2001, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2001, "gamma", "2024-04-01", "2024-05-31")

add_tag_period(2002, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2002, "delta", "2024-03-01", "2024-04-30")

add_tag_period(2003, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2003, "delta", "2024-05-01", "2024-05-31")

add_tag_period(2004, "sigma", "2024-02-01", "2024-05-31")
add_tag_period(2004, "alpha", "2024-04-01", "2024-05-31")

add_tag_period(2005, "sigma", "2024-02-01", "2024-05-31")
add_tag_period(2005, "alpha", "2024-03-01", "2024-04-30")

add_tag_period(2006, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2006, "sigma", "2024-02-01", "2024-03-31")

add_tag_period(2007, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2007, "beta",  "2024-03-01", "2024-05-31")

add_tag_period(2008, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2008, "beta",  "2024-02-01", "2024-04-30")

add_tag_period(2009, "alpha", "2024-02-01", "2024-05-31")
add_tag_period(2009, "delta", "2024-02-01", "2024-05-31")

tag_periods = pd.DataFrame(tag_period_rows)

# ============================================================
# 4) MATCHES
# ============================================================
match_rows = []
match_id = 1

def add_match(a, b):
    global match_id
    match_rows.append({"match_id": match_id, "customer_id_1": a, "customer_id_2": b})
    match_id += 1

# random background matches
for _ in range(620):
    a, b = np.random.choice(ALL_CUSTOMERS, size=2, replace=False)
    add_match(int(a), int(b))

# hand-tuned special graph
special_edges = [
    (2000, 2006),
    (2000, 2009),
    (2001, 2006),
    (2001, 2008),
    (2002, 2009),
    (2002, 2006),
    (2003, 2009),
    (2003, 2007),
    (2004, 2008),
    (2004, 2005),
    (2005, 2008),
    (2005, 2009),
    (2006, 2009),
    (2007, 2009),
]
for a, b in special_edges:
    add_match(a, b)

# add reversals and duplicates for some special edges
for a, b in [(2000, 2006), (2004, 2005), (2005, 2008), (2002, 2009)]:
    add_match(b, a)
    add_match(a, b)

matches = pd.DataFrame(match_rows)

# ============================================================
# 5) BLOCKED PAIRS
# ============================================================
blocked_rows = []

def add_block(a, b):
    blocked_rows.append({"customer_id_1": a, "customer_id_2": b})

# random blocked subset
tmp = matches[["customer_id_1", "customer_id_2"]].copy()
tmp["a"] = tmp[["customer_id_1", "customer_id_2"]].min(axis=1)
tmp["b"] = tmp[["customer_id_1", "customer_id_2"]].max(axis=1)
tmp = tmp[["a", "b"]].drop_duplicates()

random_blocked = tmp.sample(frac=0.08, random_state=42)
for _, r in random_blocked.iterrows():
    add_block(int(r["a"]), int(r["b"]))

# hand-tuned decisive block: this should hurt 2004 if applied correctly
add_block(2004, 2008)

blocked_pairs = pd.DataFrame(blocked_rows).drop_duplicates().reset_index(drop=True)

# ============================================================
# 6) MONTHLY ADJUSTMENTS
# ============================================================
# random background adjustments
monthly_adjustments_rows = []

def add_adjustment(cid, month, replacement_total):
    monthly_adjustments_rows.append({
        "customer_id": cid,
        "month": month,
        "replacement_total": round(float(replacement_total), 6)
    })

# derive candidate months from event dates
def parse_event_date_simple(s):
    s = str(s)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.NaT

tmp_events = events.copy()
tmp_events["dt"] = tmp_events["event_date"].apply(parse_event_date_simple)
tmp_events["month"] = tmp_events["dt"].dt.to_period("M").astype(str)
candidate_months = tmp_events[["customer_id", "month"]].dropna().drop_duplicates()

sample_adj = candidate_months.sample(min(120, len(candidate_months)), random_state=42)
for _, r in sample_adj.iterrows():
    add_adjustment(int(r["customer_id"]), str(r["month"]), np.random.uniform(80, 1200))

# hand-tuned decisive replacements
# 2001: replacement should replace, not add
add_adjustment(2001, "2024-03", 180.0)
# 2006: reduction to keep it close
add_adjustment(2006, "2024-04", 220.0)
# 2009: modest replacement
add_adjustment(2009, "2024-03", 310.0)
# 2005 slight boost
add_adjustment(2005, "2024-05", 520.0)

monthly_adjustments = pd.DataFrame(monthly_adjustments_rows).drop_duplicates(
    subset=["customer_id", "month"], keep="last"
).reset_index(drop=True)

# ============================================================
# 7) SAVE
# ============================================================
customers_full.to_csv("customers.csv", index=False)
events.to_csv("events.csv", index=False)
tag_periods.to_csv("tag_periods.csv", index=False)
matches.to_csv("matches.csv", index=False)
blocked_pairs.to_csv("blocked_pairs.csv", index=False)
monthly_adjustments.to_csv("monthly_adjustments.csv", index=False)

print("✅ Files created:")
for fn in [
    "customers.csv",
    "events.csv",
    "tag_periods.csv",
    "matches.csv",
    "blocked_pairs.csv",
    "monthly_adjustments.csv",
]:
    print(fn)