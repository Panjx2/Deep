Simple mixed-date benchmark package

Contents:
- prompt.txt
- generator.py
- ground_truth.py
- evaluator.py
- dataset/
    - customers.csv
    - events.csv
    - monthly_adjustments.csv
    - correct_answer.json
    - flawed_model_output_example.json
    - score_result_example.json

Design goal:
This task is intentionally simpler than the earlier 6-file network version.
The main difficulty is mixed event-date parsing plus month-level refund logic plus replacement-not-add monthly adjustments.

Why this fits the brief:
- synthetic dataset only
- no external data
- no ML training
- not a pure algorithmic puzzle
- objective answer format and objective score
