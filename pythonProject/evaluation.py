import json


def score_answer(correct_answer_file, model_answer_file):
    """
    Score the LLM's answer against the correct answer
    """
    # Load answers
    with open(correct_answer_file, 'r') as f:
        correct = json.load(f)

    with open(model_answer_file, 'r') as f:
        model = json.load(f)

    if not isinstance(model, list):
        return 0

    correct_set = set(correct)
    model_set = set(model)

    if not model_set:
        return 0

    # Jaccard similarity
    intersection = len(correct_set & model_set)
    union = len(correct_set | model_set)
    jaccard = intersection / union if union > 0 else 0

    # Penalty for wrong size (should be exactly 7)
    size_penalty = 1 - abs(len(model_set) - 7) * 0.1
    size_penalty = max(0, min(1, size_penalty))

    final_score = jaccard * size_penalty

    # Save score
    score_result = {
        "correct_answer": correct,
        "model_answer": model,
        "intersection": list(correct_set & model_set),
        "false_positives": list(model_set - correct_set),
        "false_negatives": list(correct_set - model_set),
        "jaccard_similarity": round(jaccard, 4),
        "size_penalty": round(size_penalty, 4),
        "final_score": round(final_score, 4)
    }

    with open('score_result.json', 'w') as f:
        json.dump(score_result, f, indent=2)

    print(f"Score: {final_score:.4f}")
    return final_score


# Calculate score
score = score_answer('correct_answer.json', 'model_output.json')