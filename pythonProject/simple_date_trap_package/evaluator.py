import json
import statistics
from pathlib import Path


def score_lists(correct, model):
    if not isinstance(correct, list) or not isinstance(model, list):
        return 0.0

    if len(correct) != 5:
        raise ValueError("Correct answer must contain exactly 5 IDs.")

    if len(model) != 5:
        return 0.0

    try:
        correct = [int(x) for x in correct]
        model = [int(x) for x in model]
    except Exception:
        return 0.0

    if len(set(model)) != 5:
        return 0.0

    correct_set = set(correct)
    model_set = set(model)

    overlap_score = len(correct_set & model_set) / 5.0
    position_score = sum(1 for a, b in zip(correct, model) if a == b) / 5.0

    final_score = round(0.75 * overlap_score + 0.25 * position_score, 4)

    result = {
        "correct_answer": correct,
        "model_answer": model,
        "overlap_count": len(correct_set & model_set),
        "overlap_score": round(overlap_score, 4),
        "position_score": round(position_score, 4),
        "missing_ids": sorted(list(correct_set - model_set)),
        "extra_ids": sorted(list(model_set - correct_set)),
        "final_score": final_score,
    }
    return result


def score_answer(correct_answer_file, model_answer_file, output_file="score_result.json"):
    with open(correct_answer_file, "r") as f:
        correct = json.load(f)

    with open(model_answer_file, "r") as f:
        model = json.load(f)

    result = score_lists(correct, model)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Score: {result['final_score']:.4f}")
    return result["final_score"]


def score_multiple_runs(correct_answer_file, model_answer_files, output_file="score_summary.json"):
    with open(correct_answer_file, "r") as f:
        correct = json.load(f)

    run_results = []
    scores = []

    for path in model_answer_files:
        path = str(path)
        with open(path, "r") as f:
            model = json.load(f)

        result = score_lists(correct, model)
        run_results.append({
            "file": path,
            **result
        })
        scores.append(result["final_score"])

    if not scores:
        raise ValueError("No model output files were provided.")

    summary = {
        "num_runs": len(scores),
        "scores": scores,
        "mean_score": round(statistics.mean(scores), 4),
        "median_score": round(statistics.median(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "run_results": run_results,
    }

    if len(scores) > 1:
        summary["stdev_score"] = round(statistics.pstdev(scores), 4)

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Runs:   {summary['num_runs']}")
    print(f"Mean:   {summary['mean_score']:.4f}")
    print(f"Median: {summary['median_score']:.4f}")
    print(f"Min:    {summary['min_score']:.4f}")
    print(f"Max:    {summary['max_score']:.4f}")
    if "stdev_score" in summary:
        print(f"Stdev:  {summary['stdev_score']:.4f}")

    return summary


def score_multiple_runs_from_dir(
    correct_answer_file,
    model_outputs_dir,
    pattern="model_output*.json",
    output_file="score_summary.json"
):
    model_outputs = sorted(Path(model_outputs_dir).glob(pattern))
    if not model_outputs:
        raise ValueError(
            f"No files matched pattern '{pattern}' in directory '{model_outputs_dir}'."
        )
    return score_multiple_runs(correct_answer_file, model_outputs, output_file)


if __name__ == "__main__":
    # Single run:
    # score_answer("correct_answer.json", "model_output.json")

    # Multiple runs from explicit list:
    score_multiple_runs(
        "correct_answer.json",
        ["dataset/model_output_1.json", "dataset/model_output_2.json", "dataset/model_output_3.json"]
    )

    # Multiple runs from a directory:
    # score_multiple_runs_from_dir("correct_answer.json", "runs")

    pass