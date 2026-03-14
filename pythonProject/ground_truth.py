import json
import math
import pandas as pd

def parse_mixed_date(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    return pd.NaT

def normalize_pairs(df, c1, c2):
    out = df[[c1, c2]].copy()
    out["a"] = out[[c1, c2]].min(axis=1)
    out["b"] = out[[c1, c2]].max(axis=1)
    out = out[out["a"] != out["b"]][["a", "b"]].drop_duplicates().reset_index(drop=True)
    return out

def taxed_value(amount, tax_rate):
    return float(amount) * (1.0 + float(tax_rate))

def bucket_of_value(v):
    return math.floor(v / 50.0)

# --------------------------------------------------
# Load
# --------------------------------------------------
customers = pd.read_csv("customers.csv")
events = pd.read_csv("events.csv")
tag_periods = pd.read_csv("tag_periods.csv")
matches = pd.read_csv("matches.csv")
blocked_pairs = pd.read_csv("blocked_pairs.csv")
monthly_adjustments = pd.read_csv("monthly_adjustments.csv")

# --------------------------------------------------
# Deduplicate customers
# --------------------------------------------------
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
customers = (
    customers.sort_values(["customer_id", "signup_date"])
             .drop_duplicates(subset=["customer_id"], keep="first")
             .copy()
)

# --------------------------------------------------
# Parse and clean events
# --------------------------------------------------
events["event_dt"] = events["event_date"].apply(parse_mixed_date)
events["amount"] = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0)
events["tax_rate"] = pd.to_numeric(events["tax_rate"], errors="coerce").fillna(0.0)

events = events.merge(
    customers[["customer_id", "signup_date"]],
    on="customer_id",
    how="left"
)

events = events.sort_values(["customer_id", "event_dt", "event_id"]).copy()

# --------------------------------------------------
# Stateful own monthly totals with refund-by-bucket matching
# --------------------------------------------------
monthly_rows = []

for cid, g in events.groupby("customer_id", sort=False):
    active_state = True
    signup = g["signup_date"].iloc[0]

    # month -> bucket -> stack of unmatched counted purchase values
    month_bucket_stacks = {}

    for _, row in g.iterrows():
        dt_ = row["event_dt"]
        if pd.isna(dt_):
            continue

        etype = row["event_type"]

        if etype == "pause":
            active_state = False
            continue

        if etype == "resume":
            active_state = True
            continue

        month = dt_.to_period("M").strftime("%Y-%m")

        if etype == "purchase":
            if active_state and pd.notna(signup) and dt_ >= signup:
                v = taxed_value(row["amount"], row["tax_rate"])
                b = bucket_of_value(v)
                month_bucket_stacks.setdefault(month, {}).setdefault(b, []).append(v)
            continue

        if etype == "refund":
            v = taxed_value(row["amount"], row["tax_rate"])
            b = bucket_of_value(v)
            stack = month_bucket_stacks.get(month, {}).get(b, [])
            if len(stack) > 0:
                stack.pop()
                month_bucket_stacks[month][b] = stack
            continue

    for month, bucket_map in month_bucket_stacks.items():
        total = 0.0
        for _, vals in bucket_map.items():
            total += sum(vals)
        monthly_rows.append((cid, month, total))

monthly = pd.DataFrame(monthly_rows, columns=["customer_id", "month", "computed_month_total"])
if len(monthly) == 0:
    monthly = pd.DataFrame(columns=["customer_id", "month", "computed_month_total"])
else:
    monthly = (
        monthly.groupby(["customer_id", "month"], as_index=False)["computed_month_total"]
               .sum()
    )

# --------------------------------------------------
# Replacement adjustments
# --------------------------------------------------
monthly_adjustments["replacement_total"] = pd.to_numeric(monthly_adjustments["replacement_total"], errors="coerce")

monthly = monthly.merge(
    monthly_adjustments,
    on=["customer_id", "month"],
    how="outer"
)

monthly["computed_month_total"] = monthly["computed_month_total"].fillna(0.0)
monthly["effective_month_total"] = monthly["replacement_total"].where(
    monthly["replacement_total"].notna(),
    monthly["computed_month_total"]
)

# --------------------------------------------------
# Normalize matches and remove blocked
# --------------------------------------------------
match_pairs = normalize_pairs(matches, "customer_id_1", "customer_id_2")
blocked_norm = normalize_pairs(blocked_pairs, "customer_id_1", "customer_id_2")
blocked_set = set(map(tuple, blocked_norm[["a", "b"]].to_numpy()))

match_pairs = match_pairs[
    ~match_pairs.apply(lambda r: (r["a"], r["b"]) in blocked_set, axis=1)
].copy()

edges_ab = match_pairs.rename(columns={"a": "customer_id", "b": "neighbor_id"})
edges_ba = match_pairs.rename(columns={"b": "customer_id", "a": "neighbor_id"})
edges = pd.concat([edges_ab, edges_ba], ignore_index=True)

# --------------------------------------------------
# Expand tag periods to month activity
# --------------------------------------------------
tag_periods["start_dt"] = pd.to_datetime(tag_periods["start_date"], errors="coerce")
tag_periods["end_dt"] = pd.to_datetime(tag_periods["end_date"], errors="coerce")

tag_month_rows = []
for _, row in tag_periods.iterrows():
    if pd.isna(row["start_dt"]) or pd.isna(row["end_dt"]):
        continue
    start_m = row["start_dt"].to_period("M")
    end_m = row["end_dt"].to_period("M")
    for p in pd.period_range(start_m, end_m, freq="M"):
        tag_month_rows.append((row["customer_id"], str(p), row["tag"]))

tag_months = pd.DataFrame(tag_month_rows, columns=["customer_id", "month", "tag"]).drop_duplicates()
tag_map = tag_months.groupby(["customer_id", "month"])["tag"].apply(set).to_dict()

# --------------------------------------------------
# Monthly contributions
# --------------------------------------------------
reliability = customers.set_index("customer_id")["reliability_score"].to_dict()

month_totals = monthly[["customer_id", "month", "effective_month_total"]].copy()
own_map = {
    (int(r.customer_id), str(r.month)): float(r.effective_month_total)
    for r in month_totals.itertuples(index=False)
}

months = sorted(month_totals["month"].dropna().astype(str).unique())

contrib_rows = []
for month in months:
    for _, r in edges.iterrows():
        a = int(r["customer_id"])
        b = int(r["neighbor_id"])

        # contributor reliability rule
        if reliability.get(b, 0) < 700:
            continue

        tags_a = tag_map.get((a, month), set())
        tags_b = tag_map.get((b, month), set())

        shared = tags_a & tags_b
        if len(shared) == 0:
            continue

        # reciprocal month-specific eligibility:
        # since tag-sharing is symmetric, this is effectively enforced by shared tags in this month;
        # the rule is still explicit in the prompt.
        weight = len(shared)
        val = weight * own_map.get((b, month), 0.0)
        contrib_rows.append((a, month, b, val))

monthly_contrib = pd.DataFrame(
    contrib_rows,
    columns=["customer_id", "month", "contributor_id", "weighted_monthly_contribution"]
)

if len(monthly_contrib) == 0:
    monthly_contrib = pd.DataFrame(
        columns=["customer_id", "month", "contributor_id", "weighted_monthly_contribution"]
    )

monthly_contrib = monthly_contrib.sort_values(
    ["customer_id", "month", "weighted_monthly_contribution", "contributor_id"],
    ascending=[True, True, False, True]
).copy()

monthly_contrib["rank"] = monthly_contrib.groupby(["customer_id", "month"]).cumcount() + 1
monthly_contrib = monthly_contrib[monthly_contrib["rank"] <= 2].copy()

monthly_contrib_sum = (
    monthly_contrib.groupby(["customer_id", "month"], as_index=False)["weighted_monthly_contribution"]
                  .sum()
                  .rename(columns={"weighted_monthly_contribution": "top2_neighbor_sum"})
)

# --------------------------------------------------
# Final scores
# --------------------------------------------------
customer_month = monthly[["customer_id", "month", "effective_month_total"]].copy()
customer_month = customer_month.merge(
    monthly_contrib_sum,
    on=["customer_id", "month"],
    how="left"
)
customer_month["top2_neighbor_sum"] = customer_month["top2_neighbor_sum"].fillna(0.0)
customer_month["month_score"] = (
    customer_month["effective_month_total"] + 0.2 * customer_month["top2_neighbor_sum"]
)

final_scores = (
    customer_month.groupby("customer_id", as_index=False)["month_score"]
                  .sum()
                  .rename(columns={"month_score": "final_network_score"})
)

final_scores = customers[["customer_id", "account_status"]].merge(
    final_scores,
    on="customer_id",
    how="left"
)
final_scores["final_network_score"] = final_scores["final_network_score"].fillna(0.0)

active = final_scores[final_scores["account_status"] == 0].copy()

top_5 = (
    active.sort_values(["final_network_score", "customer_id"], ascending=[False, True])
          .head(5)["customer_id"]
          .tolist()
)

with open("correct_answer.json", "w") as f:
    json.dump(top_5, f)

print(top_5)