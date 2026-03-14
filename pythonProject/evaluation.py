import json

def score_answer(correct_answer_file, model_answer_file):
    with open(correct_answer_file, "r") as f:
        correct = json.load(f)

    with open(model_answer_file, "r") as f:
        model = json.load(f)

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

    with open("score_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Score: {final_score:.4f}")
    return final_score


# Example usage
score = score_answer("correct_answer.json", "model_output.json")