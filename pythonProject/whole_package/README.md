# Structured Reasoning Stress Test for LLM Data Analysis

## Overview

This repository contains a **fully synthetic data-analysis task** designed to expose **structured reasoning failures** in strong LLMs.

The objective is simple to state: compute the **top five customers by final adjusted spend** from three CSV files.

The difficulty does **not** come from large-scale computation, machine learning, or advanced programming. Instead, it comes from the need to correctly combine several interacting data-processing rules, including:

- mixed date parsing
- customer deduplication
- chronological event handling
- month-constrained refund matching
- monthly replacement adjustments
- ranking under filtering constraints

This makes the task a useful **stress test for LLM-based analytical reasoning**: the model can often solve it **partially**, but tends to fail on one or more rules.

---

## Why this task is difficult for LLMs

The benchmark is intentionally constructed to trigger common failure modes in LLM-driven data analysis:

1. **Date normalization errors**  
   `events.csv` contains dates in three formats:
   - `YYYY-MM-DD`
   - `DD/MM/YYYY`
   - `MM-DD-YYYY`

   A correct solution must parse all of them correctly.

2. **Stateful refund logic**  
   Refunds do not subtract from totals directly.  
   They cancel the **most recent unmatched purchase** by the same customer **within the same calendar month**.

3. **Chronological reasoning**  
   Events must be processed in time order, with ties broken by `event_id`.

4. **Duplicate detection with timestamp tolerance**  
   Near-duplicate customer rows must be identified using a sub-second rule.

5. **Replacement rather than addition**  
   Monthly adjustments replace computed totals for a `(customer_id, month)` pair; they are not additive corrections.

Each individual rule is manageable for a human analyst, but combining them correctly is where LLMs often produce **partially correct answers**.

---

## Task

The model receives three CSV files:

- `customers.csv`
- `events.csv`
- `monthly_adjustments.csv`

It must compute the final ranking and output exactly:

```
[id1, id2, id3, id4, id5]
```

with **no additional text**.

The IDs must be ordered from **highest to lowest final adjusted spend**.

---

## Processing Rules

A correct solution must apply all of the following rules.

### 1. Customer deduplication

Two customer rows are duplicates if:

- all columns except `signup_date` are identical, and
- the `signup_date` values differ by **less than 1 second**

Keep the **earlier** row and discard the later one.

### 2. Active customers

A customer is considered active if:

```
account_status == 0
```

Only active customers may appear in the final ranking.

### 3. Event ordering

Events must be processed per customer in **chronological order**.

If two events have the same date, break ties using `event_id`.

### 4. Purchases

A purchase counts only if:

```
event_date >= signup_date
```

Its value is:

```
amount * (1 + tax_rate)
```

If `amount` is missing, treat it as `0`.

### 5. Refunds

Refunds **do not subtract directly from totals**.

Instead, a refund cancels the **most recent unmatched purchase** by the same customer **within the same calendar month**.

If no such purchase exists, ignore the refund.

### 6. Monthly totals

For each customer and month:

```
monthly total = sum of remaining purchases after refund cancellation
```

### 7. Monthly adjustments

If `monthly_adjustments.csv` contains a row for `(customer_id, month)`, then:

```
effective_monthly_total = replacement_total
```

This value **replaces** the computed monthly total. It is **not added** to it.

### 8. Final adjusted spend

For each customer:

```
final adjusted spend = sum of effective monthly totals across all months
```

### 9. Final ranking

Return the **top five active customers** by final adjusted spend, ordered from **highest to lowest**.

---

## Human-solvability and ambiguity

The task is designed to be **unambiguous for a careful human analyst**.

Although several rules interact, each rule is explicitly specified and deterministic:

- duplicate detection is precisely defined
- refund behavior is precisely defined
- date formats are explicitly listed
- tie-breaking is explicitly defined
- adjustment logic is explicitly defined as replacement, not addition

This keeps the task difficult for LLMs while still being **well-posed**.

---

## Dataset

The dataset is **fully synthetic** and generated locally. No external data is used.

Files:
- `dataset/customers.csv`
- `dataset/events.csv`
- `dataset/monthly_adjustments.csv`

The generator can produce the benchmark instance reproducibly.

---

## Reproducibility

### 1. Generate the dataset
```
python generator.py
```

This creates:
- `dataset/customers.csv`
- `dataset/events.csv`
- `dataset/monthly_adjustments.csv`

### 2. Compute the exact answer
```
python ground_truth.py
```

This writes:
- `correct_answer.json`
- `ground_truth_details.json`

### 3. Run the target model

Provide the model with:
- `prompt.txt`
- `dataset/customers.csv`
- `dataset/events.csv`
- `dataset/monthly_adjustments.csv`

Save each model response as a separate file in `dataset/`, for example:

```
dataset/model_output_1.json
dataset/model_output_2.json
dataset/model_output_3.json
```

### 4. Evaluate model outputs
```
python evaluator.py
```

This reads:
- `dataset/model_output_*.json`

and writes:
- `diagnostic_report_v2.json`
- `diagnostic_report_v2.txt`

---

## Expected deliverables

This repository provides the required elements of the task:

- synthetic dataset
- task prompt
- correct answer
- ground-truth Python solution
- model output
- evaluation measure

---

## Evaluation metric

Let:
- `correct` = ground-truth top-5 list
- `model` = model-predicted top-5 list

The score is:

```
score = 0.75 * overlap_score + 0.25 * position_score
```

where:

- `overlap_score = (# correct IDs appearing anywhere in model output) / 5`
- `position_score = (# IDs in the correct position) / 5`

**Interpretation:**

- `1.0` = exact match
- `0.0` = completely incorrect
- `(0, 1)` = partially correct answer

This scoring was chosen because the task output is a ranked top-5 list, and partial answers should receive credit both for:

- **identifying the correct customers**, and
- **placing them in the correct order**

---

## Ground-truth solution

The reference solution is implemented in:

```
ground_truth.py
```

It performs the full pipeline:

- load all CSV files
- deduplicate customers
- filter active customers
- normalize event dates
- sort events correctly
- apply purchase and refund rules
- compute monthly totals
- apply replacement adjustments
- rank customers by final adjusted spend

The script writes both the final answer and intermediate details for verification.

---

## Observed model behavior

The target model tested on this task was:

**GPT-5.3 Reasoning**

The task was designed so that the model would typically score **above 0 but below 1**, demonstrating partial analytical success with structured reasoning failures rather than complete failure.

### Typical failure modes include:

- misreading slash-formatted dates
- applying refunds across months
- treating adjustments as additive instead of replacement
- mishandling duplicate customer rows
- producing the correct set of IDs in the wrong order

---

## Diagnostic analysis

The evaluator does more than compute a scalar score.

It also **groups repeated incorrect outputs** and maps them to likely reasoning mistakes, making it easier to identify which rule the model likely misunderstood.

This is useful because the benchmark is not only about **whether** the model fails, but **how** it fails.

---

## Summary

This benchmark is intended to test whether an LLM can correctly integrate multiple deterministic data-analysis rules under realistic analytical constraints.

It is:

- ✅ fully synthetic
- ✅ reproducible
- ✅ unambiguous
- ✅ non-ML
- ✅ non-trivial
- ✅ designed to induce partial failure in strong models
