import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


SEED = 42


def generate_dataset(output_dir="."):
    """
    Generate a synthetic benchmark dataset with intentionally separated failure modes.

    Design goals:
    - exact semantics produce one stable top-5;
    - slash month/day confusion produces a different top-5;
    - dash day/month confusion produces a different top-5;
    - replacement-vs-add produces a different top-5;
    - refund FIFO/LIFO and same-month-vs-any-month also change the top-5;
    - active filtering, deduplication, missing amounts, and pre-signup gating all remain relevant.

    The generated files are written to:
        <output_dir>/dataset/customers.csv
        <output_dir>/dataset/events.csv
        <output_dir>/dataset/monthly_adjustments.csv
    """
    rng = np.random.default_rng(SEED)

    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # customers.csv
    # ------------------------------------------------------------------
    background_ids = list(range(1000, 1120))  # 120 low-spend background customers
    special_active_ids = list(range(2000, 2010))
    special_inactive_ids = [2200, 2201]        # high-spend decoys for active filtering
    threshold_control_ids = [2300, 2301]       # dedup-threshold controls; intentionally have no events
    base_signup = datetime(2024, 1, 1)

    customers = []
    for cid in background_ids:
        customers.append({
            "customer_id": cid,
            "reliability_score": int(rng.integers(350, 850)),
            "potential_value": int(rng.integers(20_000, 180_000)),
            "account_status": int(rng.choice([0, 1, 2], p=[0.78, 0.14, 0.08])),
            "signup_date": base_signup + timedelta(days=int(rng.integers(0, 90))),
        })

    special_signups = {
        # Main active benchmark customers.
        2000: datetime(2024, 1, 1),
        2001: datetime(2024, 1, 1),
        2002: datetime(2024, 1, 1),
        2003: datetime(2024, 1, 1),
        2004: datetime(2024, 1, 1),
        2005: datetime(2024, 3, 15),  # slash-sensitive: exact ignores one purchase, month-first counts it
        2006: datetime(2024, 3, 15),  # dash-sensitive: exact ignores one purchase, day-first counts it
        2007: datetime(2024, 1, 1),   # replacement-vs-add
        2008: datetime(2024, 1, 1),   # refund FIFO-vs-LIFO same-month
        2009: datetime(2024, 1, 1),   # missing-amount reserve / future failure-family slot
        # Inactive decoys.
        2200: datetime(2024, 1, 1),
        2201: datetime(2024, 1, 1),
        # Dedup threshold controls.
        2300: datetime(2024, 1, 1),
        2301: datetime(2024, 1, 2),
    }

    for i, cid in enumerate(special_active_ids + special_inactive_ids + threshold_control_ids):
        if cid in special_active_ids:
            status = 0
            reliability = 600 + i * 9
            potential = 120_000 + i * 4_000
        elif cid in special_inactive_ids:
            status = 1 if cid == 2200 else 2
            reliability = 600 + i * 9
            potential = 120_000 + i * 4_000
        else:
            # Threshold controls are low-impact and eventless; they exist only to exercise
            # the deduplication time-threshold logic without changing the leaderboard.
            status = 0
            reliability = 455 + (cid - 2300) * 7
            potential = 25_000 + (cid - 2300) * 2_500

        customers.append({
            "customer_id": cid,
            "reliability_score": reliability,
            "potential_value": potential,
            "account_status": status,
            "signup_date": special_signups[cid],
        })

    customers = pd.DataFrame(customers)

    # Near-duplicate rows: same non-signup fields, signup times differ by microseconds.
    duplicate_ids = [2000, 2001, 2005, 2006, 2008, 2200] + background_ids[:10]
    duplicates = customers[customers["customer_id"].isin(duplicate_ids)].copy()
    duplicates["signup_date"] = pd.to_datetime(duplicates["signup_date"]) + pd.to_timedelta(1, unit="us")
    customers_full = pd.concat([customers, duplicates], ignore_index=True)

    # Threshold controls just over the one-second boundary. These keep the SAME customer_id,
    # so they genuinely exercise the dedup rule. They are kept eventless so that, under the
    # intended semantics, they do not affect the leaderboard.
    far_duplicates = customers[customers["customer_id"].isin(threshold_control_ids)].copy()
    far_duplicates["signup_date"] = pd.to_datetime(far_duplicates["signup_date"]) + pd.to_timedelta(1100, unit="ms")
    customers_full = pd.concat([customers_full, far_duplicates], ignore_index=True)

    # ------------------------------------------------------------------
    # events.csv
    # ------------------------------------------------------------------
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

    # Background events: deliberately mixed formats, some pre-signup purchases, some refunds.
    for cid in background_ids:
        signup = pd.to_datetime(customers.loc[customers["customer_id"] == cid, "signup_date"].iloc[0])

        purchase_1 = signup + timedelta(days=int(rng.integers(1, 6)))
        purchase_2 = signup + timedelta(days=int(rng.integers(20, 35)))
        add_event(
            cid,
            "purchase",
            purchase_1.strftime("%Y-%m-%d"),
            float(rng.integers(15, 30)),
            round(float(rng.uniform(0.03, 0.09)), 6),
        )
        add_event(
            cid,
            "purchase",
            purchase_2.strftime("%m-%d-%Y"),
            float(rng.integers(18, 35)),
            round(float(rng.uniform(0.03, 0.09)), 6),
        )

        if cid % 4 == 0:
            refund_dt = signup + timedelta(days=int(rng.integers(26, 40)))
            add_event(
                cid,
                "refund",
                refund_dt.strftime("%d/%m/%Y"),
                float(rng.integers(10, 25)),
                round(float(rng.uniform(0.03, 0.09)), 6),
            )

        if cid % 7 == 0:
            pre_signup = signup - timedelta(days=int(rng.integers(2, 10)))
            add_event(
                cid,
                "purchase",
                pre_signup.strftime("%Y-%m-%d"),
                float(rng.integers(60, 120)),
                round(float(rng.uniform(0.05, 0.11)), 6),
            )

    # --------------------------------------------------------------
    # Hand-tuned active customers.
    # Exact correct top-5 under intended semantics is driven by these.
    # --------------------------------------------------------------

    # 2000: stable exact anchor.
    add_event(2000, "purchase", "2024-01-10", 800.0, 0.0)
    add_event(2000, "purchase", "2024-02-13", 800.0, 0.0)

    # 2001: cross-month refund should be ignored under exact semantics.
    add_event(2001, "purchase", "2024-01-11", 700.0, 0.0)
    add_event(2001, "purchase", "2024-02-14", 850.0, 0.0)
    add_event(2001, "refund", "2024-03-01", 0.0, 0.0)

    # 2002: direct subtraction vs cancellation matters; any-month cancellation also hurts it.
    add_event(2002, "purchase", "2024-01-12", 600.0, 0.0)
    add_event(2002, "purchase", "2024-02-15", 900.0, 0.0)
    add_event(2002, "refund", "2024-03-05", 1100.0, 0.0)

    # 2003: reserve customer, stable.
    add_event(2003, "purchase", "2024-01-13", 700.0, 0.0)
    add_event(2003, "purchase", "2024-02-16", 660.0, 0.0)

    # 2004: exact top-5 reserve boundary.
    add_event(2004, "purchase", "2024-01-14", 700.0, 0.0)
    add_event(2004, "purchase", "2024-02-17", 700.0, 0.0)

    # 2005: slash-sensitive. Exact parse: 04/03/2024 = 4 Mar < signup -> ignored.
    # Month-first parse: 04/03/2024 = 3 Apr >= signup -> counted.
    add_event(2005, "purchase", "2024-05-10", 1390.0, 0.0)
    add_event(2005, "purchase", "04/03/2024", 900.0, 0.0)

    # 2006: dash-sensitive. Exact parse: 03-04-2024 = 4 Mar < signup -> ignored.
    # Day-first parse: 03-04-2024 = 3 Apr >= signup -> counted.
    add_event(2006, "purchase", "2024-05-11", 1380.0, 0.0)
    add_event(2006, "purchase", "03-04-2024", 910.0, 0.0)

    # 2007: replacement-vs-add. Exact uses 200 (replacement) + 1170; add mode uses 900+200+1170.
    add_event(2007, "purchase", "2024-06-10", 900.0, 0.0)
    add_event(2007, "purchase", "2024-07-10", 1170.0, 0.0)

    # 2008: same-month refund cancellation. LIFO keeps 950, FIFO keeps 100.
    add_event(2008, "purchase", "2024-08-01", 950.0, 0.0)
    add_event(2008, "purchase", "2024-08-10", 100.0, 0.0)
    add_event(2008, "refund", "2024-08-20", 0.0, 0.0)
    add_event(2008, "purchase", "2024-09-05", 500.0, 0.0)

    # 2009: reserve / missing-amount slot.
    # This customer is deliberately close to the boundary so it can later be used for a
    # missing-amount failure family once the evaluator models that mistake correctly.
    add_event(2009, "purchase", "2024-01-20", 700.0, 0.0)
    add_event(2009, "purchase", "2024-02-18", 650.0, 0.0)
    add_event(2009, "purchase", "2024-03-05", np.nan, 0.2)

    # Inactive decoys: should be excluded from final ranking even though spend is huge.
    add_event(2200, "purchase", "2024-01-05", 2500.0, 0.0)
    add_event(2200, "purchase", "2024-02-05", 2400.0, 0.0)
    add_event(2201, "purchase", "2024-01-06", 2450.0, 0.0)
    add_event(2201, "purchase", "2024-02-06", 2350.0, 0.0)

    events = pd.DataFrame(event_rows)

    # Additional random missing-amount noise among the background population.
    mask = (
        events["customer_id"].isin(background_ids)
        & (events["event_type"] == "purchase")
        & (events["event_id"] % 29 == 0)
    )
    events.loc[mask, "amount"] = np.nan

    # ------------------------------------------------------------------
    # monthly_adjustments.csv
    # ------------------------------------------------------------------
    adjustment_rows = [
        # Main hand-tuned replacement row.
        {"customer_id": 2007, "month": "2024-06", "replacement_total": 200.0},
        # A few background adjustments to keep the file non-trivial.
        {"customer_id": 1001, "month": "2024-01", "replacement_total": 30.0},
        {"customer_id": 1008, "month": "2024-02", "replacement_total": 15.0},
        {"customer_id": 1017, "month": "2024-03", "replacement_total": 22.5},
    ]
    monthly_adjustments = pd.DataFrame(adjustment_rows)

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    customers_full.to_csv(os.path.join(dataset_dir, "customers.csv"), index=False)
    events.to_csv(os.path.join(dataset_dir, "events.csv"), index=False)
    monthly_adjustments.to_csv(os.path.join(dataset_dir, "monthly_adjustments.csv"), index=False)

    print("✅ Dataset created")
    print("customers.csv:", len(customers_full))
    print("events.csv:", len(events))
    print("monthly_adjustments.csv:", len(monthly_adjustments))


if __name__ == "__main__":
    generate_dataset(".")