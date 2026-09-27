"""Materialize PanNuke N10/N20 on the frozen N0/N5 zero-shot cohort.

This adds only the two missing SAM prompt budgets.  It shares the exact
manifest, DINOv2-L matcher, support references, positive prompts, and N0/N5
cohort from ``run_pannuke_zero_shot.py``.  Query GT is opened after both new
action predictions solely for scoring.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; WORKSPACE=ROOT.parent
DATA=ROOT/"data/pannuke_hf/materialized_zero_shot_v1"; MANIFEST=DATA/"manifest.csv"
OUT=ROOT/"runs/pannuke_extra_budgets_v1"; SURF=WORKSPACE/"surf"
SAM_CKPT=WORKSPACE/"assets/checkpoints/sam_vit_b.pth"; DINO_CKPT=WORKSPACE/"assets/checkpoints/dinov2_vitl14_pretrain.pth"
K=40; LAMBDA=.75
def read(path):
    with path.open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def append(path,row,fields):
    path.parent.mkdir(parents=True,exist_ok=True); header=not path.exists() or path.stat().st_size==0
    with path.open("a",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if header:w.writeheader()
        w.writerow(row)
def dice(a,b):return 2*float(np.logical_and(a,b).sum())/(float(a.sum()+b.sum())+1e-8)
def main():
    p=argparse.ArgumentParser();p.add_argument("--resume",action="store_true");p.add_argument("--max-queries",type=int,default=0);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    sys.path.insert(0,str(SURF)); from benchmarks.evaluate_negative_prompt_ablation import select_negative_points
    from benchmarks.run_monuseg_dense_prompt_experiment import IMAGE_SIZE,clean_device,compute_matching_maps_from_features,dense_points_from_score,extract_features,load_dinov2,load_mask,load_mask_image,load_sam_predictor,zscore
    rows=read(MANIFEST); queries=[x for x in rows if x['role']=='query']; supports=[x for x in rows if x['role']=='support']
    if len(queries)!=2540 or len(supports)!=5:raise RuntimeError('unexpected frozen cohort')
    if args.max_queries:queries=queries[:args.max_queries]
    progress=OUT/'per_episode.csv'; done={x['episode_id'] for x in read(progress)} if args.resume and progress.exists() else set()
    fields=['episode_id','query_case_id','support_case_id','n10_dice','n20_dice','n10_sam_score','n20_sam_score','n10_area_pct','n20_area_pct','query_gt_loaded_after_prediction','seconds']
    device=clean_device('cuda'); predictor,_=load_sam_predictor(SAM_CKPT,'vit_b','cuda'); dino=load_dinov2(DINO_CKPT,'vit_large',device)
    state=[]
    for s in supports:
        im=Image.open(DATA/s['image_path']).convert('RGB')
        with torch.inference_mode(): feat=extract_features(dino,im.resize((IMAGE_SIZE,IMAGE_SIZE),Image.BICUBIC),device)
        state.append((s,load_mask_image(DATA/s['mask_path']),feat))
    started=time.perf_counter()
    for i,q in enumerate(queries,1):
        qid=q['case_id']; im=Image.open(DATA/q['image_path']).convert('RGB'); predictor.set_image(np.asarray(im))
        with torch.inference_mode(): qfeat=extract_features(dino,im.resize((IMAGE_SIZE,IMAGE_SIZE),Image.BICUBIC),device)
        for s,smask,sfeat in state:
            eid=f"{s['case_id']}__to__{qid}"
            if eid in done:continue
            t=time.perf_counter(); fg,bg=compute_matching_maps_from_features(sfeat,smask,qfeat,device); score=zscore(fg)-LAMBDA*zscore(bg)
            pts=dense_points_from_score(score,fg,bg,im.size,max_points=K)[:K]; pos=np.asarray([[x.x,x.y] for x in pts],dtype=np.float32)
            if len(pos)!=K:raise RuntimeError(f'expected {K} positive points: {eid}')
            pred={}
            for count,name in ((10,'n10'),(20,'n20')):
                neg=select_negative_points(score,im.size,count,pos); coords=np.concatenate([pos,neg]); labels=np.concatenate([np.ones(len(pos),dtype=np.int32),np.zeros(len(neg),dtype=np.int32)])
                masks,scores,_=predictor.predict(point_coords=coords,point_labels=labels,multimask_output=True); j=int(np.argmax(scores)); mask=masks[j].astype(bool); pred[name]=(mask,float(scores[j]),100*float(mask.mean()))
            target=load_mask(DATA/q['mask_path'],im.size)
            append(progress,{'episode_id':eid,'query_case_id':qid,'support_case_id':s['case_id'],'n10_dice':dice(pred['n10'][0],target),'n20_dice':dice(pred['n20'][0],target),'n10_sam_score':pred['n10'][1],'n20_sam_score':pred['n20'][1],'n10_area_pct':pred['n10'][2],'n20_area_pct':pred['n20'][2],'query_gt_loaded_after_prediction':True,'seconds':time.perf_counter()-t},fields);done.add(eid)
        if i%10==0 or i==len(queries):print(f'[pannuke-extra] queries={i}/{len(queries)} episodes={len(done)} elapsed_s={time.perf_counter()-started:.1f}',flush=True)
    complete=len(done)==2540*5 and not args.max_queries
    (OUT/'summary.json').write_text(json.dumps({'status':'RAW_COMPLETE' if complete else 'PARTIAL','queries_completed':len({x['query_case_id'] for x in read(progress)}),'episodes_completed':len(done),'query_gt_policy':'loaded only after N10 and N20 predictions'},indent=2),encoding='utf8')
if __name__=='__main__':main()
