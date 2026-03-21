import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_LEN = 5

CONFIG = {
    "data_dir": "dataset/.",
    "outputs_glob": "model_output*.json",
    "report_json": "diagnostic_report_v2.json",
    "report_txt": "diagnostic_report_v2.txt",
}


# ============================================================
# Input parsing
# ============================================================

def safe_read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def try_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_first_int_list_from_text(text: str, expected_len: int | None = None):
    matches = re.findall(r"\[\s*-?\d+(?:\s*,\s*-?\d+){0,100}\s*\]", text)
    candidates = []
    for match in matches:
        nums = re.findall(r"-?\d+", match)
        if nums:
            candidates.append([int(x) for x in nums])

    if not candidates:
        return None

    if expected_len is not None:
        exact = [c for c in candidates if len(c) == expected_len]
        if exact:
            return exact[-1]

    return candidates[-1]


def extract_answer(obj: Any, expected_len: int | None = None):
    if isinstance(obj, list):
        try:
            return [int(x) for x in obj]
        except Exception:
            return None

    if isinstance(obj, dict):
        preferred_keys = [
            "model_answer",
            "answer",
            "output",
            "prediction",
            "result",
            "final_answer",
            "response",
            "top5",
        ]
        for key in preferred_keys:
            if key in obj:
                ans = extract_answer(obj[key], expected_len)
                if ans is not None:
                    return ans

        for value in obj.values():
            ans = extract_answer(value, expected_len)
            if ans is not None:
                return ans
        return None

    if isinstance(obj, str):
        maybe_json = try_json_loads(obj)
        if maybe_json is not None:
            ans = extract_answer(maybe_json, expected_len)
            if ans is not None:
                return ans
        return extract_first_int_list_from_text(obj, expected_len)

    return None


def load_answer_file(path: str, expected_len: int | None = None):
    try:
        text = safe_read_text(path)
    except Exception as exc:
        return None, f"failed_to_read_file: {exc}"

    parsed = try_json_loads(text)
    if parsed is not None:
        ans = extract_answer(parsed, expected_len)
        if ans is not None:
            return ans, None

    ans = extract_answer(text, expected_len)
    if ans is not None:
        return ans, None

    return None, "could_not_extract_answer"


# ============================================================
# Metrics
# ============================================================

def dcg_at_k(relevance: list[float], k: int) -> float:
    out = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        out += rel / math.log2(i + 1)
    return out



def ndcg_from_ranked_ids(correct: list[int], model: list[int], k: int) -> float:
    if not correct or not model:
        return 0.0
    ideal_rels = [1.0] * min(k, len(correct))
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    correct_set = set(correct[:k])
    rels = [1.0 if cid in correct_set else 0.0 for cid in model[:k]]
    return dcg_at_k(rels, k) / idcg



def score_lists(correct: list[int], model: list[int]) -> dict[str, Any]:
    if not isinstance(correct, list) or not isinstance(model, list):
        return {
            "valid": False,
            "error": "correct_or_model_not_a_list",
            "final_score": 0.0,
        }

    if len(correct) != EXPECTED_LEN:
        raise ValueError(f"Correct answer must contain exactly {EXPECTED_LEN} IDs.")

    try:
        correct = [int(x) for x in correct]
        model = [int(x) for x in model]
    except Exception:
        return {
            "valid": False,
            "error": "non_integer_values",
            "final_score": 0.0,
        }

    if len(model) != EXPECTED_LEN:
        return {
            "valid": False,
            "error": f"wrong_length_{len(model)}",
            "correct_answer": correct,
            "model_answer": model,
            "final_score": 0.0,
        }

    if len(set(model)) != EXPECTED_LEN:
        return {
            "valid": False,
            "error": "duplicate_ids_in_model_answer",
            "correct_answer": correct,
            "model_answer": model,
            "final_score": 0.0,
        }

    correct_set = set(correct)
    model_set = set(model)
    overlap_count = len(correct_set & model_set)
    overlap_score = overlap_count / EXPECTED_LEN
    position_hits = sum(1 for a, b in zip(correct, model) if a == b)
    position_score = position_hits / EXPECTED_LEN
    ndcg = ndcg_from_ranked_ids(correct, model, EXPECTED_LEN)
    exact_match = correct == model

    # Keep the primary score backward-compatible with the current project.
    final_score = round(0.75 * overlap_score + 0.25 * position_score, 4)

    return {
        "valid": True,
        "error": None,
        "correct_answer": correct,
        "model_answer": model,
        "exact_match": exact_match,
        "overlap_count": overlap_count,
        "overlap_score": round(overlap_score, 4),
        "position_hits": position_hits,
        "position_score": round(position_score, 4),
        "ndcg_at_5": round(ndcg, 4),
        "missing_ids": sorted(correct_set - model_set),
        "extra_ids": sorted(model_set - correct_set),
        "final_score": final_score,
    }


# ============================================================
# Scenario engine
# ============================================================

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    date_mode: str = "exact"
    adjustment_mode: str = "replace"
    refund_mode: str = "cancel_lifo"
    refund_scope: str = "same_month"
    purchase_gate: str = "on_or_after_signup"
    missing_amount_mode: str = "zero"


SCENARIOS = [
    ScenarioConfig("exact__replace"),
    ScenarioConfig("slash_monthfirst__replace", date_mode="slash_monthfirst"),
    ScenarioConfig("dash_dayfirst__replace", date_mode="dash_dayfirst"),
    ScenarioConfig("generic_dayfirst_false__replace", date_mode="generic_dayfirst_false"),
    ScenarioConfig("generic_dayfirst_true__replace", date_mode="generic_dayfirst_true"),
    ScenarioConfig("exact__add", adjustment_mode="add"),
    ScenarioConfig("slash_monthfirst__add", date_mode="slash_monthfirst", adjustment_mode="add"),
    ScenarioConfig("dash_dayfirst__add", date_mode="dash_dayfirst", adjustment_mode="add"),
    ScenarioConfig("refund_fifo_same_month__replace", refund_mode="cancel_fifo"),
    ScenarioConfig("refund_lifo_any_month__replace", refund_scope="any_month"),
    ScenarioConfig("refund_direct_subtract__replace", refund_mode="direct_subtract"),
    ScenarioConfig("ignore_signup_gate__replace", purchase_gate="ignore_signup"),
    ScenarioConfig("missing_amount_nan__replace", missing_amount_mode="nan"),
]



def parse_event_date(s: str, mode: str):
    if pd.isna(s):
        return pd.NaT
    s = str(s)

    if mode == "exact":
        fmts = ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y")
        for fmt in fmts:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.NaT

    if mode == "slash_monthfirst":
        fmts = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y")
        for fmt in fmts:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.NaT

    if mode == "dash_dayfirst":
        fmts = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
        for fmt in fmts:
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.NaT

    if mode == "generic_dayfirst_false":
        return pd.to_datetime(s, errors="coerce", dayfirst=False)

    if mode == "generic_dayfirst_true":
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

    raise ValueError(f"Unknown date parse mode: {mode}")



def deduplicate_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers["signup_dt"] = pd.to_datetime(customers["signup_date"], errors="coerce")

    group_cols = ["customer_id", "reliability_score", "potential_value", "account_status"]
    kept_rows = []

    for _, grp in customers.sort_values(group_cols + ["signup_dt"]).groupby(group_cols, sort=False):
        grp = grp.sort_values("signup_dt").reset_index(drop=True)
        keep_idx = []
        last_kept_time = None

        for i, row in grp.iterrows():
            ts = row["signup_dt"]
            if pd.isna(ts):
                continue
            if last_kept_time is None or (ts - last_kept_time).total_seconds() >= 1.0:
                keep_idx.append(i)
                last_kept_time = ts

        if keep_idx:
            kept_rows.append(grp.loc[keep_idx])

    if not kept_rows:
        return customers.iloc[0:0].copy()

    return pd.concat(kept_rows, ignore_index=True).sort_values(["customer_id", "signup_dt"]).reset_index(drop=True)



def refund_value_from_row(row) -> float:
    amount = 0.0 if pd.isna(row.amount) else float(row.amount)
    tax_rate = 0.0 if pd.isna(row.tax_rate) else float(row.tax_rate)
    return amount * (1.0 + tax_rate)



def compute_result(data_dir: str | Path, scenario: ScenarioConfig) -> dict[str, Any]:
    data_dir = Path(data_dir)

    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    monthly_adjustments = pd.read_csv(data_dir / "monthly_adjustments.csv")

    customers = deduplicate_customers(customers)

    events = events.copy()
    events["event_dt"] = events["event_date"].apply(lambda x: parse_event_date(x, scenario.date_mode))
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce")
    if scenario.missing_amount_mode == "zero":
        events["amount"] = events["amount"].fillna(0.0)
    elif scenario.missing_amount_mode == "nan":
        pass
    else:
        raise ValueError(f"Unknown missing_amount_mode: {scenario.missing_amount_mode}")

    events["tax_rate"] = pd.to_numeric(events["tax_rate"], errors="coerce").fillna(0.0)
    events = events.merge(customers[["customer_id", "signup_dt"]], on="customer_id", how="left")
    events = events.sort_values(["customer_id", "event_dt", "event_id"]).reset_index(drop=True)

    monthly_rows = []
    event_debug_rows = []

    for cid, grp in events.groupby("customer_id", sort=False):
        unmatched_by_month = defaultdict(list)
        unmatched_global = []
        month_net = defaultdict(float)

        for row in grp.itertuples(index=False):
            if pd.isna(row.event_dt):
                event_debug_rows.append({
                    "customer_id": int(row.customer_id),
                    "event_id": int(row.event_id),
                    "event_type": row.event_type,
                    "event_date_raw": row.event_date,
                    "parsed_event_dt": None,
                    "event_month": None,
                    "action": "ignored_unparsed_date",
                    "value": None,
                    "matched_purchase_event_id": None,
                })
                continue

            event_month = row.event_dt.to_period("M").strftime("%Y-%m")
            counted_purchase = (
                True if scenario.purchase_gate == "ignore_signup"
                else (pd.notna(row.signup_dt) and row.event_dt >= row.signup_dt)
            )

            if row.event_type == "purchase":
                if counted_purchase:
                    value = refund_value_from_row(row)
                    purchase_rec = {
                        "purchase_event_id": int(row.event_id),
                        "purchase_month": event_month,
                        "purchase_dt": row.event_dt,
                        "value": value,
                    }
                    unmatched_by_month[event_month].append(purchase_rec)
                    unmatched_global.append(purchase_rec)
                    month_net[event_month] += value
                    event_debug_rows.append({
                        "customer_id": int(row.customer_id),
                        "event_id": int(row.event_id),
                        "event_type": row.event_type,
                        "event_date_raw": row.event_date,
                        "parsed_event_dt": str(row.event_dt),
                        "event_month": event_month,
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
                        "event_month": event_month,
                        "action": "ignored_purchase_before_signup",
                        "value": None,
                        "matched_purchase_event_id": None,
                    })
                continue

            if row.event_type != "refund":
                event_debug_rows.append({
                    "customer_id": int(row.customer_id),
                    "event_id": int(row.event_id),
                    "event_type": row.event_type,
                    "event_date_raw": row.event_date,
                    "parsed_event_dt": str(row.event_dt),
                    "event_month": event_month,
                    "action": "ignored_unknown_event_type",
                    "value": None,
                    "matched_purchase_event_id": None,
                })
                continue

            if scenario.refund_mode == "direct_subtract":
                value = refund_value_from_row(row)
                month_net[event_month] -= value
                event_debug_rows.append({
                    "customer_id": int(row.customer_id),
                    "event_id": int(row.event_id),
                    "event_type": row.event_type,
                    "event_date_raw": row.event_date,
                    "parsed_event_dt": str(row.event_dt),
                    "event_month": event_month,
                    "action": "refund_direct_subtraction",
                    "value": value,
                    "matched_purchase_event_id": None,
                })
                continue

            if scenario.refund_scope == "same_month":
                pool = unmatched_by_month.get(event_month, [])
            elif scenario.refund_scope == "any_month":
                pool = unmatched_global
            else:
                raise ValueError(f"Unknown refund_scope: {scenario.refund_scope}")

            if not pool:
                event_debug_rows.append({
                    "customer_id": int(row.customer_id),
                    "event_id": int(row.event_id),
                    "event_type": row.event_type,
                    "event_date_raw": row.event_date,
                    "parsed_event_dt": str(row.event_dt),
                    "event_month": event_month,
                    "action": "ignored_refund_no_candidate_purchase",
                    "value": None,
                    "matched_purchase_event_id": None,
                })
                continue

            if scenario.refund_mode == "cancel_lifo":
                matched = pool.pop()
            elif scenario.refund_mode == "cancel_fifo":
                matched = pool.pop(0)
            else:
                raise ValueError(f"Unknown refund_mode: {scenario.refund_mode}")

            if scenario.refund_scope == "any_month":
                month_pool = unmatched_by_month[matched["purchase_month"]]
                for i in range(len(month_pool) - 1, -1, -1):
                    if month_pool[i]["purchase_event_id"] == matched["purchase_event_id"]:
                        month_pool.pop(i)
                        break
            else:
                # remove from global pool for consistency
                for i in range(len(unmatched_global) - 1, -1, -1):
                    if unmatched_global[i]["purchase_event_id"] == matched["purchase_event_id"]:
                        unmatched_global.pop(i)
                        break

            month_net[matched["purchase_month"]] -= matched["value"]
            event_debug_rows.append({
                "customer_id": int(row.customer_id),
                "event_id": int(row.event_id),
                "event_type": row.event_type,
                "event_date_raw": row.event_date,
                "parsed_event_dt": str(row.event_dt),
                "event_month": event_month,
                "action": "refund_cancelled_purchase",
                "value": matched["value"],
                "matched_purchase_event_id": matched["purchase_event_id"],
            })

        for month, total in month_net.items():
            monthly_rows.append((cid, month, float(total)))

    monthly = pd.DataFrame(monthly_rows, columns=["customer_id", "month", "computed_month_total"])
    if len(monthly) == 0:
        monthly = pd.DataFrame(columns=["customer_id", "month", "computed_month_total"])
    else:
        monthly = monthly.groupby(["customer_id", "month"], as_index=False)["computed_month_total"].sum()

    monthly_adjustments = monthly_adjustments.copy()
    monthly_adjustments["replacement_total"] = pd.to_numeric(monthly_adjustments["replacement_total"], errors="coerce")
    monthly = monthly.merge(monthly_adjustments, on=["customer_id", "month"], how="outer")
    monthly["computed_month_total"] = monthly["computed_month_total"].fillna(0.0)

    if scenario.adjustment_mode == "replace":
        monthly["effective_month_total"] = monthly["replacement_total"].where(
            monthly["replacement_total"].notna(),
            monthly["computed_month_total"],
        )
    elif scenario.adjustment_mode == "add":
        monthly["effective_month_total"] = monthly["computed_month_total"] + monthly["replacement_total"].fillna(0.0)
    else:
        raise ValueError(f"Unknown adjustment_mode: {scenario.adjustment_mode}")

    final = (
        monthly.groupby("customer_id", as_index=False)["effective_month_total"]
        .sum()
        .rename(columns={"effective_month_total": "final_adjusted_spend"})
    )

    final = customers[["customer_id", "account_status"]].merge(final, on="customer_id", how="left")
    final["final_adjusted_spend"] = final["final_adjusted_spend"].fillna(0.0)

    active = final[final["account_status"] == 0].copy()
    active = active.sort_values(["final_adjusted_spend", "customer_id"], ascending=[False, True]).reset_index(drop=True)
    active["rank"] = range(1, len(active) + 1)
    top5 = active.head(5)["customer_id"].tolist()

    return {
        "scenario": asdict(scenario),
        "customers": customers,
        "events": events,
        "event_debug": pd.DataFrame(event_debug_rows),
        "monthly": monthly,
        "active": active,
        "top5": top5,
    }


# ============================================================
# Diagnostics helpers
# ============================================================

def customer_rank_and_score(active_df: pd.DataFrame, customer_id: int):
    row = active_df[active_df["customer_id"] == customer_id]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return {
        "customer_id": int(customer_id),
        "rank": int(row["rank"]),
        "final_adjusted_spend": float(row["final_adjusted_spend"]),
    }



def event_month_differences(exact_events: pd.DataFrame, scenario_events: pd.DataFrame, customer_id: int):
    a = exact_events[exact_events["customer_id"] == customer_id][
        ["event_id", "event_type", "event_date", "event_dt"]
    ].copy()
    a["exact_month"] = a["event_dt"].dt.to_period("M").astype(str)
    a = a.rename(columns={"event_dt": "exact_event_dt"})

    b = scenario_events[scenario_events["customer_id"] == customer_id][["event_id", "event_dt"]].copy()
    b["scenario_month"] = b["event_dt"].dt.to_period("M").astype(str)
    b = b.rename(columns={"event_dt": "scenario_event_dt"})

    merged = a.merge(b, on="event_id", how="outer")
    diff = merged[
        (merged["exact_month"].fillna("NA") != merged["scenario_month"].fillna("NA")) |
        (merged["exact_event_dt"].fillna(pd.Timestamp.min) != merged["scenario_event_dt"].fillna(pd.Timestamp.min))
    ].copy()

    out = []
    for row in diff.sort_values("event_id").itertuples(index=False):
        out.append({
            "event_id": int(row.event_id),
            "event_type": row.event_type,
            "event_date_raw": row.event_date,
            "exact_event_dt": None if pd.isna(row.exact_event_dt) else str(row.exact_event_dt),
            "scenario_event_dt": None if pd.isna(row.scenario_event_dt) else str(row.scenario_event_dt),
            "exact_month": None if pd.isna(row.exact_month) else str(row.exact_month),
            "scenario_month": None if pd.isna(row.scenario_month) else str(row.scenario_month),
        })
    return out



def monthly_differences(exact_monthly: pd.DataFrame, scenario_monthly: pd.DataFrame, customer_id: int):
    a = exact_monthly[exact_monthly["customer_id"] == customer_id][
        ["customer_id", "month", "computed_month_total", "replacement_total", "effective_month_total"]
    ].rename(columns={
        "computed_month_total": "exact_computed",
        "replacement_total": "exact_replacement",
        "effective_month_total": "exact_effective",
    })

    b = scenario_monthly[scenario_monthly["customer_id"] == customer_id][
        ["customer_id", "month", "computed_month_total", "replacement_total", "effective_month_total"]
    ].rename(columns={
        "computed_month_total": "scenario_computed",
        "replacement_total": "scenario_replacement",
        "effective_month_total": "scenario_effective",
    })

    merged = a.merge(b, on=["customer_id", "month"], how="outer")
    for col in ["exact_computed", "scenario_computed", "exact_effective", "scenario_effective"]:
        merged[col] = merged[col].fillna(0.0)

    diff = merged[
        (merged["exact_computed"].round(9) != merged["scenario_computed"].round(9)) |
        (merged["exact_effective"].round(9) != merged["scenario_effective"].round(9))
    ].copy()

    out = []
    for row in diff.sort_values("month").itertuples(index=False):
        out.append({
            "month": str(row.month),
            "exact_computed": float(row.exact_computed),
            "scenario_computed": float(row.scenario_computed),
            "exact_effective": float(row.exact_effective),
            "scenario_effective": float(row.scenario_effective),
            "effective_delta": float(row.scenario_effective - row.exact_effective),
        })
    return out




def build_scenario_clusters(scenario_results: dict[str, dict[str, Any]]):
    grouped = defaultdict(list)
    for name, result in scenario_results.items():
        grouped[tuple(result["top5"])].append(name)

    clusters = []
    for top5, scenario_names in grouped.items():
        canonical_name = sorted(scenario_names)[0]
        clusters.append({
            "cluster_id": canonical_name,
            "scenario_names": sorted(scenario_names),
            "top5": list(top5),
            "representative_result": scenario_results[canonical_name],
        })

    clusters.sort(key=lambda x: (x["top5"], x["cluster_id"]))
    return clusters



def diagnosis_confidence_label(best_score: float) -> str:
    if best_score >= 0.95:
        return "high"
    if best_score >= 0.8:
        return "moderate"
    return "low"



def find_best_explaining_clusters(model_answer: list[int], clusters: list[dict[str, Any]]):
    scored = []
    for cluster in clusters:
        detail = score_lists(model_answer, cluster["top5"])
        scored.append({
            "cluster_id": cluster["cluster_id"],
            "scenario_names": cluster["scenario_names"],
            "top5": cluster["top5"],
            "score_vs_model": detail["final_score"],
            "detail": detail,
            "representative_result": cluster["representative_result"],
        })

    scored.sort(key=lambda x: (-x["score_vs_model"], x["cluster_id"]))
    if not scored:
        return {
            "best_clusters": [],
            "all_clusters": [],
            "best_score": 0.0,
            "cluster_level_ambiguity": False,
            "scenario_level_ambiguity": False,
            "ambiguous": False,
            "confidence_label": "low",
        }

    best_score = scored[0]["score_vs_model"]
    best_clusters = [x for x in scored if x["score_vs_model"] == best_score]
    cluster_level_ambiguity = len(best_clusters) > 1
    scenario_level_ambiguity = (len(best_clusters) == 1 and len(best_clusters[0]["scenario_names"]) > 1)

    return {
        "best_clusters": best_clusters,
        "all_clusters": scored,
        "best_score": best_score,
        "cluster_level_ambiguity": cluster_level_ambiguity,
        "scenario_level_ambiguity": scenario_level_ambiguity,
        "ambiguous": cluster_level_ambiguity or scenario_level_ambiguity,
        "confidence_label": diagnosis_confidence_label(best_score),
    }



def classify_customer_difference(
    customer_id: int,
    correct_top5: list[int],
    model_answer: list[int],
    exact_result: dict[str, Any],
    representative_result: dict[str, Any],
):
    exact_customer = customer_rank_and_score(exact_result["active"], customer_id)
    scenario_customer = customer_rank_and_score(representative_result["active"], customer_id)
    event_diffs = event_month_differences(exact_result["events"], representative_result["events"], customer_id)
    monthly_diffs = monthly_differences(exact_result["monthly"], representative_result["monthly"], customer_id)

    exact_top5_set = set(correct_top5)
    model_top5_set = set(model_answer)

    customer_changed = bool(event_diffs or monthly_diffs)
    if customer_id in exact_top5_set and customer_id not in model_top5_set:
        driver = "self_changed" if customer_changed else "displaced_by_other"
    elif customer_id in model_top5_set and customer_id not in exact_top5_set:
        driver = "self_changed" if customer_changed else "promoted_by_other"
    else:
        driver = "changed"

    opposing_ids = sorted((model_top5_set - exact_top5_set) if customer_id in exact_top5_set else (exact_top5_set - model_top5_set))
    opposing_customers = [
        customer_rank_and_score(representative_result["active"], cid) if customer_id in exact_top5_set else customer_rank_and_score(exact_result["active"], cid)
        for cid in opposing_ids
    ]
    opposing_customers = [x for x in opposing_customers if x is not None]
    opposing_customers = sorted(opposing_customers, key=lambda x: x["rank"])

    return {
        "customer": exact_customer,
        "scenario_customer": scenario_customer,
        "customer_changed": customer_changed,
        "rank_shift": None if exact_customer is None or scenario_customer is None else int(scenario_customer["rank"] - exact_customer["rank"]),
        "effective_spend_shift": None if exact_customer is None or scenario_customer is None else round(float(scenario_customer["final_adjusted_spend"] - exact_customer["final_adjusted_spend"]), 4),
        "change_driver": driver,
        "displacing_or_displaced_counterparts": opposing_customers[:3],
        "event_month_differences": event_diffs,
        "monthly_differences": monthly_diffs,
    }



def explain_missing_extra_ids(correct_top5, model_answer, exact_result, representative_result):
    correct_set = set(correct_top5)
    model_set = set(model_answer)
    missing_ids = sorted(correct_set - model_set)
    extra_ids = sorted(model_set - correct_set)

    explanation = {"missing_ids": [], "extra_ids": []}

    for cid in missing_ids:
        explanation["missing_ids"].append(
            classify_customer_difference(cid, correct_top5, model_answer, exact_result, representative_result)
        )

    for cid in extra_ids:
        explanation["extra_ids"].append(
            classify_customer_difference(cid, correct_top5, model_answer, exact_result, representative_result)
        )

    return explanation



def scenario_group_hints(names: list[str]) -> list[str]:
    hints = []
    if any("slash_monthfirst" in name for name in names):
        hints.append("slash dates likely read as month/day instead of day/month")
    if any("dash_dayfirst" in name for name in names):
        hints.append("dash dates likely read as day-month-year instead of month-day-year")
    if any(name.endswith("__add") for name in names):
        hints.append("replacement_total likely added instead of replacing computed month totals")
    if any("generic_dayfirst" in name for name in names):
        hints.append("generic parser likely used instead of explicit formats")
    if any("refund_fifo" in name for name in names):
        hints.append("refunds may be matching FIFO instead of LIFO")
    if any("refund_lifo_any_month" in name for name in names):
        hints.append("refunds may be cancelling purchases across month boundaries")
    if any("refund_direct_subtract" in name for name in names):
        hints.append("refunds may be subtracted directly instead of cancelling a prior purchase")
    if any("ignore_signup_gate" in name for name in names):
        hints.append("purchases before signup may have been counted")
    if any("missing_amount_nan" in name for name in names):
        hints.append("missing purchase amounts may not have been coerced to zero")
    return hints



def make_plain_english_diagnosis(model_answer, exact_result, cluster_match):
    score_detail = score_lists(exact_result["top5"], model_answer)
    best_clusters = cluster_match["best_clusters"]
    confidence = cluster_match.get("confidence_label", "low")

    parts = [
        (
            f"Score vs exact = {score_detail['final_score']:.4f}; "
            f"exact_match={score_detail['exact_match']}; "
            f"overlap={score_detail['overlap_count']}/5; "
            f"position_hits={score_detail['position_hits']}/5; "
            f"ndcg@5={score_detail['ndcg_at_5']:.4f}. "
            f"Diagnosis confidence: {confidence}."
        )
    ]

    if not best_clusters:
        parts.append("No diagnostic scenario cluster was available.")
        return " ".join(parts)

    if cluster_match.get("cluster_level_ambiguity"):
        cluster_summaries = [
            {"scenario_names": c["scenario_names"], "top5": c["top5"]}
            for c in best_clusters
        ]
        parts.append(
            "Diagnosis is ambiguous across multiple equally good output clusters: "
            + json.dumps(cluster_summaries)
        )
        union_hints = []
        seen = set()
        for cluster in best_clusters:
            for hint in scenario_group_hints(cluster["scenario_names"]):
                if hint not in seen:
                    seen.add(hint)
                    union_hints.append(hint)
        if union_hints:
            parts.append("Shared plausible issue(s): " + "; ".join(union_hints) + ".")
    else:
        cluster = best_clusters[0]
        if cluster_match.get("scenario_level_ambiguity"):
            parts.append(
                f"Best output cluster matched with top5 {cluster['top5']}. "
                f"Within that cluster, the answer is equally consistent with scenarios {cluster['scenario_names']}."
            )
        else:
            parts.append(
                f"Closest built-in explanation cluster: {cluster['scenario_names']} "
                f"with top5 {cluster['top5']}."
            )
        hints = scenario_group_hints(cluster["scenario_names"])
        if hints:
            prefix = "Plausible issue(s): " if confidence == "low" else "Likely issue(s): "
            parts.append(prefix + "; ".join(hints) + ".")

    if confidence == "low":
        parts.append("Treat this diagnosis as a partial resemblance rather than a definitive failure-mode attribution.")

    if score_detail["missing_ids"]:
        parts.append(f"Missing correct IDs: {score_detail['missing_ids']}.")
    if score_detail["extra_ids"]:
        parts.append(f"Unexpected extra IDs: {score_detail['extra_ids']}.")

    return " ".join(parts)



def run_diagnosis(
    data_dir: str | Path = CONFIG["data_dir"],
    outputs_glob: str = CONFIG["outputs_glob"],
    report_json: str = CONFIG["report_json"],
    report_txt: str = CONFIG["report_txt"],
):
    data_dir = Path(data_dir)
    output_files = sorted(data_dir.glob(outputs_glob))
    if not output_files:
        raise ValueError(f"No model outputs matched {outputs_glob} in {data_dir}")

    scenario_results = {scenario.name: compute_result(data_dir, scenario) for scenario in SCENARIOS}
    exact_result = scenario_results["exact__replace"]
    correct_top5 = exact_result["top5"]

    non_exact_scenario_results = {
        name: result
        for name, result in scenario_results.items()
        if name != "exact__replace"
    }
    clusters = build_scenario_clusters(non_exact_scenario_results)
    diagnostic_clusters = [c for c in clusters if c["top5"] != correct_top5]
    non_discriminative_clusters = [c for c in clusters if c["top5"] == correct_top5]

    loaded_runs = []
    scores = []
    unique_answers = Counter()

    for path in output_files:
        model_answer, err = load_answer_file(str(path), expected_len=EXPECTED_LEN)
        if err is not None or model_answer is None:
            result = {
                "file": str(path),
                "valid": False,
                "error": err,
                "correct_answer": correct_top5,
                "model_answer": None,
                "final_score": 0.0,
                "exact_match": False,
            }
        else:
            scored = score_lists(correct_top5, model_answer)
            result = {"file": str(path), **scored}
            unique_answers[tuple(model_answer)] += 1
        loaded_runs.append(result)
        scores.append(result["final_score"])

    summary = {
        "num_runs": len(scores),
        "scores": scores,
        "mean_score": round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "stdev_score": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "exact_match_rate": round(sum(1 for x in loaded_runs if x.get("exact_match")) / len(loaded_runs), 4),
        "num_unique_answers": len(unique_answers),
        "most_common_answer": list(unique_answers.most_common(1)[0][0]) if unique_answers else None,
        "exact_top5": correct_top5,
        "num_diagnostic_clusters": len(diagnostic_clusters),
        "num_non_discriminative_clusters": len(non_discriminative_clusters),
        "run_results": loaded_runs,
    }

    diagnoses = []
    for answer_tuple, count in unique_answers.items():
        model_answer = list(answer_tuple)
        exact_score = score_lists(correct_top5, model_answer)
        if exact_score["exact_match"]:
            cluster_match = {"ambiguous": False, "best_clusters": [], "all_clusters": [], "best_score": 1.0}
            representative_result = exact_result
            explanation = {"missing_ids": [], "extra_ids": []}
            plain = (
                "Exact match to the ground truth. No failure-mode diagnosis is needed for this answer."
            )
        else:
            cluster_match = find_best_explaining_clusters(model_answer, diagnostic_clusters)
            representative_result = cluster_match["best_clusters"][0]["representative_result"] if cluster_match["best_clusters"] else exact_result
            explanation = explain_missing_extra_ids(correct_top5, model_answer, exact_result, representative_result)
            plain = make_plain_english_diagnosis(model_answer, exact_result, cluster_match)

        diagnoses.append({
            "model_answer": model_answer,
            "frequency": count,
            "score_vs_exact": exact_score,
            "diagnosis_ambiguous": cluster_match["ambiguous"],
            "diagnosis_confidence": cluster_match.get("confidence_label"),
            "diagnosis_best_score": cluster_match.get("best_score"),
            "cluster_level_ambiguity": cluster_match.get("cluster_level_ambiguity", False),
            "scenario_level_ambiguity": cluster_match.get("scenario_level_ambiguity", False),
            "best_explaining_clusters": [
                {
                    "scenario_names": c["scenario_names"],
                    "top5": c["top5"],
                    "score_vs_model": c["score_vs_model"],
                    "score_detail": c["detail"],
                }
                for c in cluster_match["best_clusters"]
            ],
            "all_cluster_matches": [
                {
                    "scenario_names": c["scenario_names"],
                    "top5": c["top5"],
                    "score_vs_model": c["score_vs_model"],
                    "score_detail": c["detail"],
                }
                for c in cluster_match["all_clusters"]
            ],
            "explanation": explanation,
            "plain_english_diagnosis": plain,
        })

    scenario_catalog = [
        {
            "scenario_name": name,
            "top5": result["top5"],
            "config": result["scenario"],
        }
        for name, result in sorted(scenario_results.items())
    ]

    report = {
        "summary": summary,
        "scenario_catalog": scenario_catalog,
        "non_discriminative_clusters": [
            {
                "scenario_names": c["scenario_names"],
                "top5": c["top5"],
            }
            for c in non_discriminative_clusters
        ],
        "diagnoses": diagnoses,
    }

    Path(report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    txt_lines = []
    txt_lines.append(f"Runs:   {summary['num_runs']}")
    txt_lines.append(f"Mean:   {summary['mean_score']:.4f}")
    txt_lines.append(f"Median: {summary['median_score']:.4f}")
    txt_lines.append(f"Min:    {summary['min_score']:.4f}")
    txt_lines.append(f"Max:    {summary['max_score']:.4f}")
    txt_lines.append(f"Stdev:  {summary['stdev_score']:.4f}")
    txt_lines.append(f"Exact-match rate: {summary['exact_match_rate']:.4f}")
    txt_lines.append(f"Unique answers:   {summary['num_unique_answers']}")
    txt_lines.append(f"Exact top 5: {summary['exact_top5']}")
    txt_lines.append("")

    for idx, diag in enumerate(diagnoses, start=1):
        txt_lines.append(f"--- Diagnosis #{idx} ---")
        txt_lines.append(f"Model answer: {diag['model_answer']}")
        txt_lines.append(f"Frequency: {diag['frequency']}")
        txt_lines.append(f"Score vs exact: {diag['score_vs_exact']['final_score']:.4f}")
        txt_lines.append(f"Ambiguous diagnosis: {diag['diagnosis_ambiguous']}")
        txt_lines.append(f"Diagnosis confidence: {diag.get('diagnosis_confidence')}")
        txt_lines.append(f"Best scenario-match score: {diag.get('diagnosis_best_score')}")
        txt_lines.append(diag['plain_english_diagnosis'])
        txt_lines.append("")

    Path(report_txt).write_text("\n".join(txt_lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_diagnosis()