"""Aggregate completed five-fold FoB diagnostic without altering its protocol."""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_risk_controlled_response_gate import bootstrap_delta, write_csv

def read(p):
    with open(p, newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    root=Path('icassp/runs'); records=[]; folds=[]
    for f in range(5):
        summary=json.loads((root/f'fob_monuseg_fold{f}/test/summary.json').read_text(encoding='utf8'))
        rows=read(root/f'fob_monuseg_fold{f}/test/per_episode.csv')
        if len(rows)!=summary['episodes']: raise AssertionError(f'fold {f} count mismatch')
        for row in rows: records.append({'fold':f, **row, 'dice':float(row['dice'])})
        folds.append({'fold':f, **summary})
    by_q=defaultdict(list)
    for row in records: by_q[row['query_case_id']].append(row)
    if len(by_q)!=37 or any(len(v)!=5 for v in by_q.values()): raise AssertionError('Expected 37 queries x five supports')
    queries=[{'query_case_id':q,'references':5,'fob_dice':sum(x['dice'] for x in v)/5} for q,v in sorted(by_q.items())]
    write_csv(root/'fob_monuseg_nestedcv_v1_episodes.csv',records); write_csv(root/'fob_monuseg_nestedcv_v1_queries.csv',queries)
    result={'protocol':'fob_monuseg_nestedcv_v1','status':'COMPLETE','folds':folds,'episodes':len(records),'queries':len(queries),
            'mean_dice':sum(x['fob_dice'] for x in queries)/len(queries),'query_dice_bootstrap':bootstrap_delta([{**x,'zero':0.0} for x in queries],'fob_dice','zero'),
            'limitation':'Diagnostic only: FoB uses SAM-H and a different prompt representation; it is not protocol-matched to the DINOv2-L/SAM-B response policy.'}
    (root/'fob_monuseg_nestedcv_v1_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
