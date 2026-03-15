import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_event_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.NaT


def deduplicate_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers["signup_dt"] = pd.to_datetime(customers["signup_date"], errors="coerce")

    group_cols = ["customer_id", "reliability_score", "potential_value", "account_status"]
    kept_rows = []

    for _, g in customers.sort_values(group_cols + ["signup_dt"]).groupby(group_cols, sort=False):
        g = g.sort_values("signup_dt").reset_index(drop=True)
        keep_idx = []
        last_kept_time = None

        for i, row in g.iterrows():
            ts = row["signup_dt"]
            if pd.isna(ts):
                continue
            if last_kept_time is None or (ts - last_kept_time).total_seconds() >= 1.0:
                keep_idx.append(i)
                last_kept_time = ts

        if keep_idx:
            kept_rows.append(g.loc[keep_idx])

    if not kept_rows:
        return customers.iloc[0:0].copy()

    out = pd.concat(kept_rows, ignore_index=True)
    return out.sort_values(["customer_id", "signup_dt"]).reset_index(drop=True)


def compute_ground_truth(data_dir="dataset", special_ids=None):
    if special_ids is None:
        special_ids = [2000, 2001, 2002, 2003, 2004, 2005]

    data_dir = Path(data_dir)

    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    monthly_adjustments = pd.read_csv(data_dir / "monthly_adjustments.csv")

    customers = deduplicate_customers(customers)

    events = events.copy()
    events["event_dt"] = events["event_date"].apply(parse_event_date)
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0)
    events["tax_rate"] = pd.to_numeric(events["tax_rate"], errors="coerce").fillna(0.0)

    events = events.merge(
        customers[["customer_id", "signup_dt"]],
        on="customer_id",
        how="left",
    )

    events = events.sort_values(["customer_id", "event_dt", "event_id"]).reset_index(drop=True)

    monthly_rows = []
    event_debug_rows = []

    for cid, grp in events.groupby("customer_id", sort=False):
        unmatched = defaultdict(list)

        for row in grp.itertuples(index=False):
            if pd.isna(row.event_dt):
                event_debug_rows.append({
                    "customer_id": int(row.customer_id),
                    "event_id": int(row.event_id),
                    "event_type": row.event_type,
                    "event_date_raw": row.event_date,
                    "parsed_event_dt": None,
                    "month": None,
                    "action": "ignored_unparsed_date",
                    "value": None,
                    "matched_purchase_event_id": None,
                })
                continue

            month = row.event_dt.to_period("M").strftime("%Y-%m")

            if row.event_type == "purchase":
                if pd.notna(row.signup_dt) and row.event_dt >= row.signup_dt:
                    value = float(row.amount) * (1.0 + float(row.tax_rate))
                    unmatched[month].append({
                        "purchase_event_id": int(row.event_id),
                        "purchase_date": str(row.event_dt),
                        "value": value,
                    })
                    event_debug_rows.append({
                        "customer_id": int(row.customer_id),
                        "event_id": int(row.event_id),
                        "event_type": row.event_type,
                        "event_date_raw": row.event_date,
                        "parsed_event_dt": str(row.event_dt),
                        "month": month,
                        "action": "counted_purchase",
                        "value": value,
                        "matched_purchase_event_id": None,
                    })
                else:
                    event_debug_rows.append({
                        "customer_id": int(row.customer_id),
                        "event_id": int(row.event_id),
                        "event_type": row.event_type,
                        "event_date_raw": row.event_date,
                        "parsed_event_dt": str(row.event_dt),
                        "month": month,
                        "action": "ignored_purchase_before_signup",
                        "value": None,
                        "matched_purchase_event_id": None,
                    })

            elif row.event_type == "refund":
                if unmatched.get(month):
                    matched = unmatched[month].pop()
                    event_debug_rows.append({
                        "customer_id": int(row.customer_id),
                        "event_id": int(row.event_id),
                        "event_type": row.event_type,
                        "event_date_raw": row.event_date,
                        "parsed_event_dt": str(row.event_dt),
                        "month": month,
                        "action": "refund_cancelled_purchase",
                        "value": matched["value"],
                        "matched_purchase_event_id": matched["purchase_event_id"],
                    })
                else:
                    event_debug_rows.append({
                        "customer_id": int(row.customer_id),
                        "event_id": int(row.event_id),
                        "event_type": row.event_type,
                        "event_date_raw": row.event_date,
                        "parsed_event_dt": str(row.event_dt),
                        "month": month,
                        "action": "ignored_refund_no_unmatched_purchase",
                        "value": None,
                        "matched_purchase_event_id": None,
                    })

        for month, vals in unmatched.items():
            monthly_rows.append((cid, month, float(sum(v["value"] for v in vals))))

    monthly = pd.DataFrame(monthly_rows, columns=["customer_id", "month", "computed_month_total"])
    if len(monthly) == 0:
        monthly = pd.DataFrame(columns=["customer_id", "month", "computed_month_total"])
    else:
        monthly = (
            monthly.groupby(["customer_id", "month"], as_index=False)["computed_month_total"]
            .sum()
        )

    monthly_adjustments["replacement_total"] = pd.to_numeric(
        monthly_adjustments["replacement_total"], errors="coerce"
    )

    monthly = monthly.merge(
        monthly_adjustments,
        on=["customer_id", "month"],
        how="outer",
    )

    monthly["computed_month_total"] = monthly["computed_month_total"].fillna(0.0)
    monthly["effective_month_total"] = monthly["replacement_total"].where(
        monthly["replacement_total"].notna(),
        monthly["computed_month_total"],
    )
    monthly["used_adjustment"] = monthly["replacement_total"].notna()

    final = (
        monthly.groupby("customer_id", as_index=False)["effective_month_total"]
        .sum()
        .rename(columns={"effective_month_total": "final_adjusted_spend"})
    )

    final = customers[["customer_id", "account_status"]].merge(
        final,
        on="customer_id",
        how="left",
    )
    final["final_adjusted_spend"] = final["final_adjusted_spend"].fillna(0.0)

    active = final[final["account_status"] == 0].copy()
    active = active.sort_values(["final_adjusted_spend", "customer_id"], ascending=[False, True]).reset_index(drop=True)
    active["rank"] = range(1, len(active) + 1)

    top5 = active.head(5)["customer_id"].tolist()

    special_summary = []
    for cid in special_ids:
        rank_row = active[active["customer_id"] == cid]
        if len(rank_row) == 0:
            continue
        rank_row = rank_row.iloc[0]

        m = monthly[monthly["customer_id"] == cid].copy()
        m = m.sort_values("month")

        special_summary.append({
            "customer_id": int(cid),
            "rank": int(rank_row["rank"]),
            "final_adjusted_spend": float(rank_row["final_adjusted_spend"]),
            "monthly_breakdown": [
                {
                    "month": str(r["month"]),
                    "computed_month_total": float(r["computed_month_total"]),
                    "replacement_total": None if pd.isna(r["replacement_total"]) else float(r["replacement_total"]),
                    "effective_month_total": float(r["effective_month_total"]),
                    "used_adjustment": bool(r["used_adjustment"]),
                }
                for _, r in m.iterrows()
            ],
        })

    top5_explanations = []
    for cid in top5:
        row = active[active["customer_id"] == cid].iloc[0]
        m = monthly[monthly["customer_id"] == cid].sort_values("month")

        explanation = {
            "customer_id": int(cid),
            "rank": int(row["rank"]),
            "final_adjusted_spend": float(row["final_adjusted_spend"]),
            "why_it_is_in_top5": [
                {
                    "month": str(r["month"]),
                    "computed_month_total": float(r["computed_month_total"]),
                    "replacement_total": None if pd.isna(r["replacement_total"]) else float(r["replacement_total"]),
                    "effective_month_total": float(r["effective_month_total"]),
                    "used_adjustment": bool(r["used_adjustment"]),
                }
                for _, r in m.iterrows()
            ],
        }
        top5_explanations.append(explanation)

    event_debug = pd.DataFrame(event_debug_rows)
    special_event_debug = {}
    for cid in special_ids:
        df = event_debug[event_debug["customer_id"] == cid].copy()
        if len(df):
            special_event_debug[str(cid)] = df.sort_values("event_id").to_dict(orient="records")

    details = {
        "top5": top5,
        "top10_active_leaderboard": active.head(10)[
            ["customer_id", "rank", "final_adjusted_spend"]
        ].to_dict(orient="records"),
        "special_customer_summary": special_summary,
        "top5_explanations": top5_explanations,
        "special_event_debug": special_event_debug,
    }

    return top5, details


if __name__ == "__main__":
    top5, details = compute_ground_truth("dataset")

    with open("correct_answer.json", "w") as f:
        json.dump(top5, f, indent=2)

    with open("ground_truth_details.json", "w") as f:
        json.dump(details, f, indent=2)

    print("Correct answer:", top5)
    print("Saved:")
    print("- correct_answer.json")
    print("- ground_truth_details.json")