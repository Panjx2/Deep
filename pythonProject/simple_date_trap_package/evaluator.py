import json
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd


EXPECTED_LEN = 5

# ============================================================
# ONE-CLICK CONFIG
# Edit these if you want, then just press Run.
# ============================================================

CONFIG = {
    "data_dir": "dataset",
    "outputs_glob": "model_output*.json",
    "report_json": "diagnostic_report.json",
    "report_txt": "diagnostic_report.txt",
}


# ============================================================
# Parsing / loading helpers
# ============================================================

def safe_read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def try_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_first_int_list_from_text(text: str, expected_len: int | None = None):
    matches = re.findall(r"\[\s*-?\d+(?:\s*,\s*-?\d+){0,50}\s*\]", text)
    candidates = []
    for m in matches:
        nums = re.findall(r"-?\d+", m)
        if nums:
            candidates.append([int(x) for x in nums])

    if not candidates:
        return None

    if expected_len is not None:
        exact = [c for c in candidates if len(c) == expected_len]
        if exact:
            return exact[-1]

    return candidates[-1]


def extract_answer(obj, expected_len: int | None = None):
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
        ]
        for key in preferred_keys:
            if key in obj:
                ans = extract_answer(obj[key], expected_len)
                if ans is not None:
                    return ans

        for v in obj.values():
            ans = extract_answer(v, expected_len)
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
    except Exception as e:
        return None, f"failed_to_read_file: {e}"

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
# Scoring
# ============================================================

def score_lists(correct, model):
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

    overlap_score = len(correct_set & model_set) / EXPECTED_LEN
    position_score = sum(1 for a, b in zip(correct, model) if a == b) / EXPECTED_LEN
    final_score = round(0.75 * overlap_score + 0.25 * position_score, 4)

    return {
        "valid": True,
        "error": None,
        "correct_answer": correct,
        "model_answer": model,
        "overlap_count": len(correct_set & model_set),
        "overlap_score": round(overlap_score, 4),
        "position_score": round(position_score, 4),
        "missing_ids": sorted(list(correct_set - model_set)),
        "extra_ids": sorted(list(model_set - correct_set)),
        "final_score": final_score,
    }


# ============================================================
# Ground truth / counterfactual solvers
# ============================================================

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


def compute_result(
    data_dir: str,
    date_mode: str = "exact",
    adjustment_mode: str = "replace",
):
    data_dir = Path(data_dir)

    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    monthly_adjustments = pd.read_csv(data_dir / "monthly_adjustments.csv")

    customers = deduplicate_customers(customers)

    events = events.copy()
    events["event_dt"] = events["event_date"].apply(lambda x: parse_event_date(x, date_mode))
    events["amount"] = pd.to_numeric(events["amount"], errors="coerce").fillna(0.0)
    events["tax_rate"] = pd.to_numeric(events["tax_rate"], errors="coerce").fillna(0.0)

    events = events.merge(
        customers[["customer_id", "signup_dt"]],
        on="customer_id",
        how="left",
    )
    events["event_month"] = events["event_dt"].dt.to_period("M").astype(str)
    events = events.sort_values(["customer_id", "event_dt", "event_id"]).reset_index(drop=True)

    monthly_rows = []
    for cid, grp in events.groupby("customer_id", sort=False):
        unmatched = defaultdict(list)

        for row in grp.itertuples(index=False):
            if pd.isna(row.event_dt):
                continue

            month = row.event_dt.to_period("M").strftime("%Y-%m")

            if row.event_type == "purchase":
                if pd.notna(row.signup_dt) and row.event_dt >= row.signup_dt:
                    value = float(row.amount) * (1.0 + float(row.tax_rate))
                    unmatched[month].append({
                        "event_id": row.event_id,
                        "event_dt": row.event_dt,
                        "event_date_raw": row.event_date,
                        "value": value,
                    })

            elif row.event_type == "refund":
                if unmatched.get(month):
                    unmatched[month].pop()

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

    if adjustment_mode == "replace":
        monthly["effective_month_total"] = monthly["replacement_total"].where(
            monthly["replacement_total"].notna(),
            monthly["computed_month_total"],
        )
    elif adjustment_mode == "add":
        monthly["effective_month_total"] = (
            monthly["computed_month_total"] + monthly["replacement_total"].fillna(0.0)
        )
    else:
        raise ValueError(f"Unknown adjustment mode: {adjustment_mode}")

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
    active = active.sort_values(
        ["final_adjusted_spend", "customer_id"],
        ascending=[False, True]
    ).reset_index(drop=True)
    active["rank"] = range(1, len(active) + 1)

    top5 = active.head(5)["customer_id"].tolist()

    return {
        "date_mode": date_mode,
        "adjustment_mode": adjustment_mode,
        "customers": customers,
        "events": events,
        "monthly": monthly,
        "active": active,
        "top5": top5,
    }


# ============================================================
# Diagnostics
# ============================================================

def find_best_explaining_scenario(model_answer, scenario_results):
    scored = []
    for name, result in scenario_results.items():
        s = score_lists(model_answer, result["top5"])
        scored.append((name, s["final_score"], s, result))

    scored.sort(key=lambda x: (-x[1], x[0]))
    best_name, best_score, best_score_detail, best_result = scored[0]
    return {
        "scenario_name": best_name,
        "scenario_score_vs_model": best_score,
        "scenario_score_detail": best_score_detail,
        "scenario_result": best_result,
        "all_scenarios": [
            {
                "scenario_name": name,
                "score_vs_model": score,
                "detail": detail,
            }
            for name, score, detail, _ in scored
        ],
    }


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
        ["event_id", "event_type", "event_date", "event_dt", "event_month"]
    ].rename(columns={"event_dt": "exact_event_dt", "event_month": "exact_month"})
    b = scenario_events[scenario_events["customer_id"] == customer_id][
        ["event_id", "event_dt", "event_month"]
    ].rename(columns={"event_dt": "scenario_event_dt", "event_month": "scenario_month"})

    merged = a.merge(b, on="event_id", how="outer")
    diff = merged[
        (merged["exact_month"].fillna("NA") != merged["scenario_month"].fillna("NA")) |
        (merged["exact_event_dt"].fillna(pd.Timestamp.min) != merged["scenario_event_dt"].fillna(pd.Timestamp.min))
    ].copy()

    if len(diff) == 0:
        return []

    out = []
    for r in diff.sort_values("event_id").itertuples(index=False):
        out.append({
            "event_id": int(r.event_id),
            "event_type": r.event_type,
            "event_date_raw": r.event_date,
            "exact_event_dt": None if pd.isna(r.exact_event_dt) else str(r.exact_event_dt),
            "scenario_event_dt": None if pd.isna(r.scenario_event_dt) else str(r.scenario_event_dt),
            "exact_month": None if pd.isna(r.exact_month) else str(r.exact_month),
            "scenario_month": None if pd.isna(r.scenario_month) else str(r.scenario_month),
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
    merged["exact_computed"] = merged["exact_computed"].fillna(0.0)
    merged["scenario_computed"] = merged["scenario_computed"].fillna(0.0)
    merged["exact_effective"] = merged["exact_effective"].fillna(0.0)
    merged["scenario_effective"] = merged["scenario_effective"].fillna(0.0)

    diff = merged[
        (merged["exact_computed"].round(9) != merged["scenario_computed"].round(9)) |
        (merged["exact_effective"].round(9) != merged["scenario_effective"].round(9))
    ].copy()

    if len(diff) == 0:
        return []

    out = []
    for r in diff.sort_values("month").itertuples(index=False):
        out.append({
            "month": str(r.month),
            "exact_computed": float(r.exact_computed),
            "scenario_computed": float(r.scenario_computed),
            "exact_effective": float(r.exact_effective),
            "scenario_effective": float(r.scenario_effective),
            "effective_delta": float(r.scenario_effective - r.exact_effective),
        })
    return out


def explain_missing_extra_ids(correct_top5, model_answer, exact_result, best_result):
    correct_set = set(correct_top5)
    model_set = set(model_answer)

    missing_ids = sorted(list(correct_set - model_set))
    extra_ids = sorted(list(model_set - correct_set))

    explanation = {
        "missing_ids": [],
        "extra_ids": [],
    }

    for cid in missing_ids:
        explanation["missing_ids"].append({
            "customer": customer_rank_and_score(exact_result["active"], cid),
            "best_scenario_customer": customer_rank_and_score(best_result["active"], cid),
            "event_month_differences": event_month_differences(
                exact_result["events"], best_result["events"], cid
            ),
            "monthly_differences": monthly_differences(
                exact_result["monthly"], best_result["monthly"], cid
            ),
        })

    for cid in extra_ids:
        explanation["extra_ids"].append({
            "customer": customer_rank_and_score(exact_result["active"], cid),
            "best_scenario_customer": customer_rank_and_score(best_result["active"], cid),
            "event_month_differences": event_month_differences(
                exact_result["events"], best_result["events"], cid
            ),
            "monthly_differences": monthly_differences(
                exact_result["monthly"], best_result["monthly"], cid
            ),
        })

    return explanation


def make_plain_english_diagnosis(model_answer, exact_result, best_match):
    best_name = best_match["scenario_name"]
    best_result = best_match["scenario_result"]
    correct_top5 = exact_result["top5"]

    details = score_lists(correct_top5, model_answer)
    missing_ids = details.get("missing_ids", [])
    extra_ids = details.get("extra_ids", [])

    parts = []
    parts.append(
        f"The model output matches the exact top 5 with score {details['final_score']:.4f} "
        f"(overlap {details['overlap_count']}/5, position score {details['position_score']:.1f})."
    )

    parts.append(
        f"The closest built-in failure mode is '{best_name}', whose own top 5 best matches the model output."
    )

    if best_name == "slash_monthfirst__replace":
        parts.append(
            "This usually means slash dates were interpreted as month/day instead of day/month. "
            "That moves some purchases into the wrong months, so refunds cancel different purchases and "
            "monthly adjustment replacement applies to different computed totals."
        )
    elif best_name == "dash_dayfirst__replace":
        parts.append(
            "This usually means dash dates were interpreted as day-month-year instead of month-day-year."
        )
    elif best_name.endswith("__add"):
        parts.append(
            "This usually means replacement totals were added to computed monthly totals instead of replacing them."
        )
    elif "generic_dayfirst" in best_name:
        parts.append(
            "This usually means a generic date parser was used instead of explicit format handling."
        )

    if missing_ids:
        parts.append(f"Missing correct IDs: {missing_ids}.")
    if extra_ids:
        parts.append(f"Unexpected extra IDs: {extra_ids}.")

    if 2005 in missing_ids:
        parts.append(
            "Customer 2005 is omitted because one or more of its load-bearing event dates likely moved to the wrong month "
            "under the best-matching failure mode. That changes which purchases remain unmatched after refunds, so 2005’s "
            "monthly totals drop and it falls below the cutoff."
        )

    return " ".join(parts)


# ============================================================
# Main runner
# ============================================================

def run_diagnosis(
    data_dir="dataset",
    outputs_glob="model_output*.json",
    report_json="diagnostic_report.json",
    report_txt="diagnostic_report.txt",
):
    data_dir = Path(data_dir)
    output_files = sorted(data_dir.glob(outputs_glob))
    if not output_files:
        raise ValueError(f"No model outputs matched {outputs_glob} in {data_dir}")

    exact_result = compute_result(data_dir, date_mode="exact", adjustment_mode="replace")
    correct_top5 = exact_result["top5"]

    scenario_specs = {
        "exact__replace": ("exact", "replace"),
        "slash_monthfirst__replace": ("slash_monthfirst", "replace"),
        "dash_dayfirst__replace": ("dash_dayfirst", "replace"),
        "generic_dayfirst_false__replace": ("generic_dayfirst_false", "replace"),
        "generic_dayfirst_true__replace": ("generic_dayfirst_true", "replace"),
        "exact__add": ("exact", "add"),
        "slash_monthfirst__add": ("slash_monthfirst", "add"),
    }

    scenario_results = {
        name: compute_result(data_dir, date_mode=dmode, adjustment_mode=amode)
        for name, (dmode, amode) in scenario_specs.items()
    }

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
        "exact_top5": correct_top5,
        "run_results": loaded_runs,
    }

    diagnoses = []
    for answer_tuple, count in unique_answers.items():
        model_answer = list(answer_tuple)
        best_match = find_best_explaining_scenario(
            model_answer,
            {k: v for k, v in scenario_results.items() if k != "exact__replace"}
        )

        explanation = explain_missing_extra_ids(
            correct_top5=correct_top5,
            model_answer=model_answer,
            exact_result=exact_result,
            best_result=best_match["scenario_result"],
        )

        plain = make_plain_english_diagnosis(
            model_answer=model_answer,
            exact_result=exact_result,
            best_match=best_match,
        )

        diagnoses.append({
            "model_answer": model_answer,
            "frequency": count,
            "score_vs_exact": score_lists(correct_top5, model_answer),
            "best_match_scenario": best_match["scenario_name"],
            "best_match_score_vs_model": best_match["scenario_score_vs_model"],
            "best_match_top5": best_match["scenario_result"]["top5"],
            "all_scenario_matches": best_match["all_scenarios"],
            "explanation": explanation,
            "plain_english_diagnosis": plain,
        })

    report = {
        "summary": summary,
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
    txt_lines.append("")
    txt_lines.append(f"Exact top 5: {correct_top5}")
    txt_lines.append("")

    for i, d in enumerate(diagnoses, start=1):
        txt_lines.append(f"--- Diagnosis #{i} ---")
        txt_lines.append(f"Model answer: {d['model_answer']}")
        txt_lines.append(f"Frequency: {d['frequency']}")
        txt_lines.append(f"Score vs exact: {d['score_vs_exact']['final_score']:.4f}")
        txt_lines.append(f"Best matching failure mode: {d['best_match_scenario']}")
        txt_lines.append(f"Best matching failure-mode top 5: {d['best_match_top5']}")
        txt_lines.append(d["plain_english_diagnosis"])
        txt_lines.append("")

        for miss in d["explanation"]["missing_ids"]:
            customer = miss.get("customer")
            if customer and customer["customer_id"] == 2005:
                txt_lines.append("Focused note on missing customer 2005:")
                txt_lines.append(json.dumps(miss, indent=2))
                txt_lines.append("")
                break

    Path(report_txt).write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"Runs:   {summary['num_runs']}")
    print(f"Mean:   {summary['mean_score']:.4f}")
    print(f"Median: {summary['median_score']:.4f}")
    print(f"Min:    {summary['min_score']:.4f}")
    print(f"Max:    {summary['max_score']:.4f}")
    print(f"Stdev:  {summary['stdev_score']:.4f}")
    print(f"Exact top 5: {correct_top5}")
    print(f"Saved JSON report to: {report_json}")
    print(f"Saved text report to: {report_txt}")

    return report


if __name__ == "__main__":
    run_diagnosis()