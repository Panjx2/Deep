import csv
import json
import random
from pathlib import Path
from statistics import median

random.seed(2026)
units = [f"U{i}" for i in range(1,9)]
days = [f"2026-02-{d:02d}" for d in range(1,11)]
shifts = ['dawn','day','swing','night']

rows = []
for ds in days:
    for u in units:
        shift = shifts[(sum(ord(c) for c in (u + ds)) % 4)]
        items_in = random.randint(180, 339)
        scrap = random.randint(8, 39)
        items_out = items_in - scrap
        rework = random.randint(0, 25)
        downtime = random.randint(0, 35)
        energy = float(random.randint(55, 119))
        quality = 'ok' if random.random() < 0.78 else 'review'
        r = random.random()
        flow = 'stable' if r < 0.70 else ('spike' if r < 0.85 else 'dip')
        rows.append({
            'date': ds, 'unit_id': u, 'shift': shift, 'items_in': items_in, 'items_out': items_out,
            'rework_minutes': rework, 'downtime_minutes': downtime, 'energy_kwh': energy,
            'quality_flag': quality, 'flow_signal': flow,
        })

special = {
    ('2026-02-03','U3'): {'quality_flag':'review', 'downtime_minutes':34, 'flow_signal':'spike'},
    ('2026-02-04','U6'): {'rework_minutes':25, 'flow_signal':'dip'},
    ('2026-02-06','U1'): {'quality_flag':'review', 'rework_minutes':22},
    ('2026-02-07','U8'): {'downtime_minutes':35, 'flow_signal':'spike'},
    ('2026-02-09','U4'): {'energy_kwh':118.0, 'flow_signal':'dip'},
}
for row in rows:
    key = (row['date'], row['unit_id'])
    if key in special:
        row.update(special[key])

quality_map = {'ok':1.0, 'review':0.91}
shift_map = {'dawn':1.02,'day':1.00,'swing':0.99,'night':0.97}
flow_map = {'stable':0.0,'spike':-6.0,'dip':-3.5}

for row in rows:
    row_eff = (
        row['items_out'] * quality_map[row['quality_flag']]
        - 0.75 * row['rework_minutes']
        - 1.15 * row['downtime_minutes']
        + flow_map[row['flow_signal']]
    )
    row['row_index'] = (row_eff / row['energy_kwh']) * shift_map[row['shift']]

vals = [r['row_index'] for r in rows]
med = median(vals)
mad = median([abs(v - med) for v in vals])
lower = med - 2.8*mad
upper = med + 2.8*mad
for row in rows:
    row['clipped_index'] = min(upper, max(lower, row['row_index']))

unit_first = {u: [] for u in units}
unit_second = {u: [] for u in units}
for row in rows:
    if row['date'] <= '2026-02-05':
        unit_first[row['unit_id']].append(row['clipped_index'])
    else:
        unit_second[row['unit_id']].append(row['clipped_index'])

unit_score = {}
for u in units:
    m1 = sum(unit_first[u])/len(unit_first[u])
    m2 = sum(unit_second[u])/len(unit_second[u])
    unit_score[u] = 0.6*m1 + 0.4*m2

ranked_units = [k for k,_ in sorted(unit_score.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

# daily averages
by_day = {}
for d in days:
    day_vals = [r['clipped_index'] for r in rows if r['date']==d]
    by_day[d] = sum(day_vals)/len(day_vals)

sorted_daily = sorted(by_day.values())
# linear interpolation quantile q=0.10 on n points
q = 0.10
n = len(sorted_daily)
pos = (n - 1) * q
lo = int(pos)
hi = min(lo + 1, n - 1)
frac = pos - lo
threshold = sorted_daily[lo] * (1 - frac) + sorted_daily[hi] * frac

alert_days = sorted([d for d,v in by_day.items() if v < threshold])
checksum = int(round(sum(unit_score[u] for u in ranked_units) * 1000)) + 17*len(alert_days)

out = {'ranked_units': ranked_units, 'alert_days': alert_days, 'checksum': checksum}

root = Path('/workspace/Deep/recruitment_task')
root.mkdir(exist_ok=True)

fieldnames = ['date','unit_id','shift','items_in','items_out','rework_minutes','downtime_minutes','energy_kwh','quality_flag','flow_signal']
with (root/'synthetic_factory_log.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k:r[k] for k in fieldnames})

(root/'task_prompt.txt').write_text("""You are given a synthetic dataset in `synthetic_factory_log.csv`.
Use only this file (no external data). Execute Python code to solve the task.

Task
1) Load the dataset.
2) For each row compute `row_index` using:
   row_eff = (items_out * quality_multiplier) - (0.75*rework_minutes) - (1.15*downtime_minutes) + flow_bonus
   row_index = (row_eff / energy_kwh) * shift_multiplier

   Mapping tables:
   quality_multiplier: ok->1.00, review->0.91
   shift_multiplier: dawn->1.02, day->1.00, swing->0.99, night->0.97
   flow_bonus: stable->0.0, spike->-6.0, dip->-3.5

3) Winsorize `row_index` globally using median ± 2.8*MAD (median absolute deviation).
   Call the clipped value `clipped_index`.

4) Compute each unit's final score:
   final_score = 0.6 * mean(clipped_index for 2026-02-01..2026-02-05)
               + 0.4 * mean(clipped_index for 2026-02-06..2026-02-10)

5) Let `ranked_units` be top 3 unit_id values by `final_score` descending.
   Tie-breaker (if needed): alphabetical unit_id.

6) For each date compute average `clipped_index` across all units.
   Define `alert_days` as dates whose daily average is STRICTLY below the 10th percentile of daily averages
   (linear interpolation).

7) Compute:
   checksum = round(sum(final_score of top 3 units)*1000) + 17*len(alert_days)

Output format (stdout, and nothing else):
{"ranked_units":[...],"alert_days":[...],"checksum":number}
""")

(root/'correct_answer.json').write_text(json.dumps(out, separators=(',',':')))

(root/'ground_truth_solution.py').write_text("""import csv
import json
from statistics import median

rows = []
with open('synthetic_factory_log.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['items_out'] = float(row['items_out'])
        row['rework_minutes'] = float(row['rework_minutes'])
        row['downtime_minutes'] = float(row['downtime_minutes'])
        row['energy_kwh'] = float(row['energy_kwh'])
        rows.append(row)

quality = {'ok':1.00, 'review':0.91}
shift = {'dawn':1.02, 'day':1.00, 'swing':0.99, 'night':0.97}
flow = {'stable':0.0, 'spike':-6.0, 'dip':-3.5}
for r in rows:
    row_eff = (r['items_out'] * quality[r['quality_flag']]) - 0.75*r['rework_minutes'] - 1.15*r['downtime_minutes'] + flow[r['flow_signal']]
    r['row_index'] = (row_eff / r['energy_kwh']) * shift[r['shift']]

vals = [r['row_index'] for r in rows]
med = median(vals)
mad = median([abs(v-med) for v in vals])
low = med - 2.8*mad
high = med + 2.8*mad
for r in rows:
    r['clipped_index'] = min(high, max(low, r['row_index']))

units = sorted({r['unit_id'] for r in rows})
scores = {}
for u in units:
    first = [r['clipped_index'] for r in rows if r['unit_id']==u and r['date'] <= '2026-02-05']
    second = [r['clipped_index'] for r in rows if r['unit_id']==u and r['date'] >= '2026-02-06']
    scores[u] = 0.6*(sum(first)/len(first)) + 0.4*(sum(second)/len(second))

ranked = [u for u,_ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]

dates = sorted({r['date'] for r in rows})
daily = []
for d in dates:
    vals = [r['clipped_index'] for r in rows if r['date']==d]
    daily.append((d, sum(vals)/len(vals)))

ordered = sorted(v for _,v in daily)
q = 0.10
n = len(ordered)
pos = (n-1)*q
lo = int(pos)
hi = min(lo+1, n-1)
frac = pos-lo
threshold = ordered[lo]*(1-frac) + ordered[hi]*frac
alerts = sorted([d for d,v in daily if v < threshold])
checksum = int(round(sum(scores[u] for u in ranked) * 1000)) + 17*len(alerts)

print(json.dumps({'ranked_units': ranked, 'alert_days': alerts, 'checksum': checksum}, separators=(',',':')))
""")

model_output = {'ranked_units': ['U8','U6','U1'], 'alert_days': alert_days, 'checksum': checksum + 41}
(root/'model_output.json').write_text(json.dumps(model_output, separators=(',',':')))

(root/'measure_definition.txt').write_text("""Scoring function S in [0,1]

Given ground truth GT and prediction P:
1) Ranking score:
   S_rank = (# exact position matches in ranked_units) / 3
2) Alert-day score:
   S_alert = F1 score between sets of alert_days
3) Checksum score:
   S_check = max(0, 1 - min(1, abs(P_checksum - GT_checksum)/200))

Final score:
S = 0.5*S_rank + 0.3*S_alert + 0.2*S_check
""")

s_rank = sum(1 for a,b in zip(model_output['ranked_units'], ranked_units) if a==b)/3
pa, ga = set(model_output['alert_days']), set(alert_days)
if not pa and not ga:
    s_alert = 1.0
else:
    inter = len(pa & ga)
    precision = inter/len(pa) if pa else 0.0
    recall = inter/len(ga) if ga else 0.0
    s_alert = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
s_check = max(0.0, 1 - min(1.0, abs(model_output['checksum'] - checksum)/200))
score = 0.5*s_rank + 0.3*s_alert + 0.2*s_check
(root/'model_score.txt').write_text(f"{score:.6f}\n")

print(json.dumps({'ground_truth': out, 'model_score': score}, indent=2))
