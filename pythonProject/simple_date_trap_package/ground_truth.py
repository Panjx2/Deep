
import json
import math
import re
import pandas as pd

def parse_event_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return pd.to_datetime(s, format='%Y-%m-%d')
    if re.match(r'^\d{2}/\d{2}/\d{4}$', s):
        return pd.to_datetime(s, format='%d/%m/%Y')
    if re.match(r'^\d{2}-\d{2}-\d{4}$', s):
        return pd.to_datetime(s, format='%m-%d-%Y')
    return pd.NaT

def solve(customers_path="dataset/customers.csv", events_path="dataset/events.csv", adjustments_path="dataset/monthly_adjustments.csv"):
    customers = pd.read_csv(customers_path)
    events = pd.read_csv(events_path)
    adjustments = pd.read_csv(adjustments_path)

    # Deduplicate near-identical customer rows
    customers["signup_dt"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers["signup_sec"] = customers["signup_dt"].dt.floor("s")
    customers = (
        customers.sort_values(["customer_id", "signup_dt"])
                 .drop_duplicates(
                     subset=["customer_id", "reliability_score", "potential_value", "account_status", "signup_sec"],
                     keep="first",
                 )
                 .drop(columns=["signup_sec"])
                 .copy()
    )

    # Parse events
    events["event_dt"] = events["event_date"].apply(parse_event_date)
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0)
    events["tax_rate"] = pd.to_numeric(events["tax_rate"], errors="coerce").fillna(0.0)

    events = events.merge(
        customers[["customer_id", "signup_dt"]],
        on="customer_id",
        how="left"
    )

    events = events.sort_values(["customer_id", "event_dt", "event_id"]).copy()

    # Monthly totals with refund cancellation
    monthly_rows = []
    from collections import defaultdict

    for customer_id, g in events.groupby("customer_id", sort=False):
        signup_dt = g["signup_dt"].iloc[0]
        unmatched = defaultdict(list)  # month -> stack of counted purchases

        for row in g.itertuples(index=False):
            if pd.isna(row.event_dt):
                continue

            month = row.event_dt.to_period("M").strftime("%Y-%m")

            if row.event_type == "purchase":
                if row.event_dt >= signup_dt:
                    value = float(row.amount) * (1.0 + float(row.tax_rate))
                    unmatched[month].append(value)

            elif row.event_type == "refund":
                if unmatched[month]:
                    unmatched[month].pop()

        for month, values in unmatched.items():
            monthly_rows.append((customer_id, month, sum(values)))

    monthly = pd.DataFrame(monthly_rows, columns=["customer_id", "month", "computed_month_total"])
    if len(monthly) > 0:
        monthly = (
            monthly.groupby(["customer_id", "month"], as_index=False)["computed_month_total"]
                   .sum()
        )
    else:
        monthly = pd.DataFrame(columns=["customer_id", "month", "computed_month_total"])

    # Replacement, not addition
    adjustments["replacement_total"] = pd.to_numeric(adjustments["replacement_total"], errors="coerce")
    monthly = monthly.merge(adjustments, on=["customer_id", "month"], how="outer")
    monthly["computed_month_total"] = monthly["computed_month_total"].fillna(0.0)
    monthly["effective_month_total"] = monthly["replacement_total"].where(
        monthly["replacement_total"].notna(),
        monthly["computed_month_total"]
    )

    final_scores = (
        monthly.groupby("customer_id", as_index=False)["effective_month_total"]
              .sum()
              .rename(columns={"effective_month_total": "final_adjusted_spend"})
    )

    final_scores = customers[["customer_id", "account_status"]].merge(
        final_scores, on="customer_id", how="left"
    )
    final_scores["final_adjusted_spend"] = final_scores["final_adjusted_spend"].fillna(0.0)

    active = final_scores[final_scores["account_status"] == 0].copy()
    active = active.sort_values(["final_adjusted_spend", "customer_id"], ascending=[False, True])

    answer = active.head(5)["customer_id"].tolist()
    return answer, active

if __name__ == "__main__":
    answer, _ = solve("dataset/customers.csv", "dataset/events.csv", "dataset/monthly_adjustments.csv")

    with open("correct_answer.json", "w") as f:
        json.dump(answer, f)

    print(answer)
