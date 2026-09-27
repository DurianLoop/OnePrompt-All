"""Collect independent ExtraTrees-seed repeats for the four-action policy."""
from __future__ import annotations
import json
from pathlib import Path
def main():
    sources={"20260715":Path('icassp/runs/multibudget_negative_policy_v1/summary.json'),
             **{str(s):Path(f'icassp/runs/multibudget_seed_stability_v1/seed_{s}/summary.json') for s in (20260716,20260717,20260718)}}
    rows=[]
    for seed,path in sources.items():
        x=json.loads(path.read_text(encoding='utf8'))['four_action_vs_two_action']
        rows.append({'extra_trees_seed':int(seed),**x})
    result={'protocol':'four_action_seed_stability_v1','status':'COMPLETE','repeats':rows,
      'all_ci_lower_positive':all(x['ci95_low']>0 for x in rows),
      'mean_delta_range':[min(x['mean_delta'] for x in rows),max(x['mean_delta'] for x in rows)],
      'scope':'Only ExtraTrees random seed varies; splits, raw DINO/SAM outcomes, action space, and selection procedure are fixed.'}
    out=Path('icassp/runs/multibudget_seed_stability_v1/summary.json');out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
