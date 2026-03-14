import csv
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
