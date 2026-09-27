"""Aggregate the frozen PanNuke N0/N5/N10/N20 fixed-budget ablation."""
from __future__ import annotations
import csv,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent)); from run_risk_controlled_response_gate import bootstrap_delta,write_csv
def read(p):
    with open(p,newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def main():
    base={x['episode_id']:x for x in read('icassp/runs/pannuke_zero_shot_v1/per_episode.csv')}; extra={x['episode_id']:x for x in read('icassp/runs/pannuke_extra_budgets_v1/per_episode.csv')}
    if set(base)!=set(extra):raise RuntimeError(f'incomplete/mismatched cohorts: base={len(base)} extra={len(extra)}')
    byq=defaultdict(list)
    for eid,x in base.items():
        e=extra[eid]; byq[x['query_case_id']].append({'query_case_id':x['query_case_id'],'n0_dice':float(x['n0_dice']),'n5_dice':float(x['n5_dice']),'n10_dice':float(e['n10_dice']),'n20_dice':float(e['n20_dice'])})
    if len(byq)!=2540 or any(len(v)!=5 for v in byq.values()):raise RuntimeError('expected 2540 queries x five references')
    rows=[]
    for q,v in sorted(byq.items()):rows.append({'query_case_id':q,'references':5,**{a:sum(x[a] for x in v)/5 for a in ('n0_dice','n5_dice','n10_dice','n20_dice')},'oracle_dice':sum(max(x[a] for a in ('n0_dice','n5_dice','n10_dice','n20_dice')) for x in v)/5})
    out=Path('icassp/runs/pannuke_extra_budgets_v1');write_csv(out/'per_query_four_budget.csv',rows)
    means={a:sum(x[a] for x in rows)/len(rows) for a in ('n0_dice','n5_dice','n10_dice','n20_dice')}; best=max(means,key=means.get)
    result={'protocol':'pannuke_four_fixed_budget_v1','status':'COMPLETE','queries':len(rows),'supports_per_query':5,'means':means,'best_fixed':best,'n10_vs_n0':bootstrap_delta(rows,'n10_dice','n0_dice'),'n20_vs_n0':bootstrap_delta(rows,'n20_dice','n0_dice'),'oracle_minus_best_fixed':bootstrap_delta(rows,'oracle_dice',best),'note':'Fixed-budget ablation only; oracle is an unattainable diagnostic ceiling.'}
    (out/'four_budget_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
