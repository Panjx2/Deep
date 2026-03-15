
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dataset(output_dir="."):
    np.random.seed(42)

    random_ids = list(range(1000, 1150))   # 150 background customers
    special_ids = list(range(2000, 2006))  # 6 hand-tuned customers
    base_signup = datetime(2024, 1, 1)

    # -----------------------------
    # customers.csv
    # -----------------------------
    customers = []
    for cid in random_ids:
        customers.append({
            "customer_id": cid,
            "reliability_score": np.random.randint(300, 850),
            "potential_value": np.random.randint(20000, 200000),
            "account_status": np.random.choice([0, 1, 2], p=[0.72, 0.18, 0.10]),
            "signup_date": base_signup + timedelta(days=int(np.random.randint(0, 120))),
        })

    special_signup = {cid: datetime(2024, 2, 1) + timedelta(days=cid - 2000) for cid in special_ids}
    for i, cid in enumerate(special_ids):
        customers.append({
            "customer_id": cid,
            "reliability_score": 500 + i * 20,
            "potential_value": 120000 + i * 5000,
            "account_status": 0,
            "signup_date": special_signup[cid],
        })

    customers = pd.DataFrame(customers)

    # Near-duplicate rows differing only by microseconds in signup_date
    dup_ids = random_ids[:12] + special_ids[:4]
    duplicates = customers[customers["customer_id"].isin(dup_ids)].copy()
    duplicates["signup_date"] = duplicates["signup_date"] + pd.to_timedelta(1, unit="us")
    customers_full = pd.concat([customers, duplicates], ignore_index=True)

    # -----------------------------
    # events.csv
    # -----------------------------
    event_rows = []
    event_id = 1

    def add_event(customer_id, event_type, event_date, amount, tax_rate):
        nonlocal event_id
        event_rows.append({
            "event_id": event_id,
            "customer_id": customer_id,
            "event_type": event_type,
            "amount": amount,
            "tax_rate": tax_rate,
            "event_date": event_date,
        })
        event_id += 1

    def fmt_date(dt, i):
        if i % 3 == 0:
            return dt.strftime("%Y-%m-%d")
        elif i % 3 == 1:
            return dt.strftime("%d/%m/%Y")
        else:
            return dt.strftime("%m-%d-%Y")

    # Background random events
    for cid in random_ids:
        signup = customers.loc[customers["customer_id"] == cid, "signup_date"].iloc[0]
        n_events = np.random.randint(6, 11)

        dates = []
        for i in range(n_events):
            if i < 2 and np.random.rand() < 0.2:
                dt = signup - timedelta(days=int(np.random.randint(1, 15)))
            else:
                dt = signup + timedelta(days=int(np.random.randint(0, 210)))
            dates.append(dt)

        dates = sorted(dates)
        for i, dt in enumerate(dates):
            if np.random.rand() < 0.82:
                add_event(
                    cid,
                    "purchase",
                    fmt_date(dt, i),
                    float(np.random.randint(20, 120)),
                    round(0.08 + np.random.random() * 0.05, 6),
                )
            else:
                add_event(
                    cid,
                    "refund",
                    fmt_date(dt, i),
                    float(np.random.randint(20, 120)),
                    round(0.08 + np.random.random() * 0.05, 6),
                )

    # Hand-tuned special customers
    # The crux is mixed date parsing + month assignment + replacement-not-add adjustments.
    special_events = {
        2000: [
            ("purchase", "03/04/2024", 400.0, 0.10),
            ("purchase", "04-15-2024", 300.0, 0.10),
            ("refund",   "25/04/2024", 290.0, 0.10),
            ("purchase", "05/04/2024", 350.0, 0.10),
            ("purchase", "2024-05-08", 450.0, 0.10),
        ],
        2001: [
            ("purchase", "04/05/2024", 420.0, 0.10),
            ("purchase", "05-18-2024", 320.0, 0.10),
            ("refund",   "20/05/2024", 300.0, 0.10),
            ("purchase", "06/05/2024", 360.0, 0.10),
            ("purchase", "2024-06-11", 430.0, 0.10),
        ],
        2002: [
            ("purchase", "05/06/2024", 390.0, 0.10),
            ("purchase", "06-17-2024", 310.0, 0.10),
            ("refund",   "25/06/2024", 290.0, 0.10),
            ("purchase", "07/06/2024", 340.0, 0.10),
            ("purchase", "2024-07-12", 410.0, 0.10),
        ],
        2003: [
            ("purchase", "06/07/2024", 380.0, 0.10),
            ("purchase", "07-14-2024", 330.0, 0.10),
            ("refund",   "20/07/2024", 310.0, 0.10),
            ("purchase", "08/07/2024", 350.0, 0.10),
            ("purchase", "2024-08-12", 420.0, 0.10),
        ],
        2004: [
            ("purchase", "07/08/2024", 360.0, 0.10),
            ("purchase", "08-21-2024", 320.0, 0.10),
            ("refund",   "27/08/2024", 300.0, 0.10),
            ("purchase", "09/08/2024", 340.0, 0.10),
            ("purchase", "2024-09-10", 405.0, 0.10),
        ],
        2005: [
            ("purchase", "08/09/2024", 390.0, 0.10),
            ("purchase", "09-19-2024", 520.0, 0.10),
            ("refund",   "24/09/2024", 280.0, 0.10),
            ("purchase", "10/09/2024", 335.0, 0.10),
            ("purchase", "2024-10-11", 398.0, 0.10),
            ("purchase", "2024-10-25", 350.0, 0.10),
        ],
    }

    for cid, evs in special_events.items():
        for event_type, event_date, amount, tax_rate in evs:
            add_event(cid, event_type, event_date, amount, tax_rate)

    events = pd.DataFrame(event_rows)

    # Missing amount trap
    mask = (events.index % 37 == 0) & (events["event_type"] == "purchase")
    events.loc[mask, "amount"] = np.nan

    # -----------------------------
    # monthly_adjustments.csv
    # -----------------------------
    def parse_event_date_for_generation(s):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.NaT

    temp = events.copy()
    temp["dt"] = temp["event_date"].map(parse_event_date_for_generation)
    temp["month"] = temp["dt"].dt.to_period("M").astype(str)
    candidate_months = temp[["customer_id", "month"]].dropna().drop_duplicates()

    sample_adj = candidate_months.sample(min(40, len(candidate_months)), random_state=42)
    adjustment_rows = []
    for _, r in sample_adj.iterrows():
        adjustment_rows.append({
            "customer_id": int(r["customer_id"]),
            "month": str(r["month"]),
            "replacement_total": round(float(np.random.uniform(60, 350)), 6),
        })

    # Hand-tuned replacements
    adjustment_rows += [
        {"customer_id": 2000, "month": "2024-04", "replacement_total": 650.0},
        {"customer_id": 2001, "month": "2024-05", "replacement_total": 700.0},
        {"customer_id": 2002, "month": "2024-06", "replacement_total": 690.0},
        {"customer_id": 2003, "month": "2024-07", "replacement_total": 710.0},
        {"customer_id": 2004, "month": "2024-08", "replacement_total": 705.0},
        # Intentionally no adjustment for 2005 so the date parsing matters more.
    ]

    monthly_adjustments = pd.DataFrame(adjustment_rows).drop_duplicates(
        subset=["customer_id", "month"], keep="last"
    )

    os.makedirs(output_dir, exist_ok=True)
    customers_full.to_csv(os.path.join(output_dir, "dataset/customers.csv"), index=False)
    events.to_csv(os.path.join(output_dir, "dataset/events.csv"), index=False)
    monthly_adjustments.to_csv(os.path.join(output_dir, "dataset/monthly_adjustments.csv"), index=False)

    print("✅ Dataset created")
    print("customers.csv:", len(customers_full))
    print("events.csv:", len(events))
    print("monthly_adjustments.csv:", len(monthly_adjustments))

if __name__ == "__main__":
    generate_dataset(".")
