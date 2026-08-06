"""Training-free same-image exemplar discovery baselines."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
from PIL import Image
from scipy import ndimage, signal
from skimage.color import rgb2lab

Algorithm = Literal["proposal", "appearance", "template"]

@dataclass
class InstanceCandidate:
    mask: np.ndarray
    score: float
    method: str
    @property
    def box(self):
        ys, xs = np.where(self.mask)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32) if len(xs) else np.zeros(4, np.float32)
    @property
    def point(self):
        ys, xs = np.where(self.mask)
        return np.array([xs.mean(), ys.mean()], np.float32) if len(xs) else np.zeros(2, np.float32)

def discover_instances(image: np.ndarray, exemplar_mask: np.ndarray, algorithm: Algorithm, proposals: Iterable[dict] | None = None, max_instances: int = 12) -> list[InstanceCandidate]:
    exemplar = np.asarray(exemplar_mask, bool)
    if exemplar.ndim != 2 or not exemplar.any(): raise ValueError("same-image discovery requires a non-empty exemplar mask")
    if algorithm == "proposal": found = _proposal(image, exemplar, proposals or [])
    elif algorithm == "appearance": found = _appearance(image, exemplar)
    elif algorithm == "template": found = _template(image, exemplar)
    else: raise ValueError(f"unknown same-image algorithm: {algorithm}")
    # Template correlation is a deliberately high-precision baseline rather
    # than an object counter. Cap it conservatively so repeated texture does
    # not turn one object into an unbounded list of scale-shifted detections.
    limit = min(max_instances, 4) if algorithm == "template" else max_instances
    return _nms(found + [InstanceCandidate(exemplar, 1.0, "exemplar")], limit)

def _lab(image): return ndimage.gaussian_filter(rgb2lab(image.astype(np.uint8)).astype(np.float32), sigma=(1.2,1.2,0))

def _resize(array, width, height, nearest=False):
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize((width, height), mode))

def _appearance(image, exemplar):
    feat = _lab(image); pixels = feat[exemplar]; mean = pixels.mean(0)
    scale = np.maximum(np.median(np.abs(pixels - np.median(pixels, 0)), 0) * 1.4826, 5.0)
    dist = np.sqrt(np.mean(((feat - mean) / scale) ** 2, 2)); threshold = float(np.clip(np.quantile(dist[exemplar], .92) * 1.75, .65, 2.8))
    binary = ndimage.binary_opening(dist <= threshold, structure=np.ones((3,3))); binary = ndimage.binary_closing(binary, structure=np.ones((5,5)))
    labels, count = ndimage.label(binary, structure=np.ones((3,3))); ref_area = max(int(exemplar.sum()), 1); out=[]
    for i in range(1, count + 1):
        area=int((labels == i).sum())
        if max(20, ref_area//20) <= area <= ref_area*5:
            mask=labels==i; out.append(InstanceCandidate(mask, float(np.exp(-dist[mask].mean())*min(area,ref_area)/max(area,ref_area)), "appearance"))
    return out

def _proposal(image, exemplar, proposals):
    feat=_lab(image); ref=feat[exemplar].mean(0); ref_area=max(float(exemplar.sum()),1.); out=[]
    for item in proposals:
        mask=np.asarray(item.get("segmentation", []), bool)
        if mask.shape != exemplar.shape or not mask.any(): continue
        area=float(mask.sum())
        if not max(24.,ref_area*.04) <= area <= ref_area*7.: continue
        colour=float(np.exp(-2.4*np.linalg.norm(feat[mask].mean(0)-ref)/180.)); shape=float(np.exp(-abs(np.log(area/ref_area)))); quality=float(item.get("predicted_iou",.7))*float(item.get("stability_score",.8))
        out.append(InstanceCandidate(mask, .58*colour+.24*shape+.18*quality, "sam-proposal"))
    return out

def _template(image, exemplar):
    ys,xs=np.where(exemplar); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1; gray=image.astype(np.float32).mean(2); patch=gray[y0:y1,x0:x1]; patch_mask=exemplar[y0:y1,x0:x1]
    if min(patch.shape)<8: return []
    peaks=[]
    for scale in (.70,.85,1.,1.18,1.35):
        width,height=max(8,int(patch.shape[1]*scale)),max(8,int(patch.shape[0]*scale))
        if width>=gray.shape[1] or height>=gray.shape[0]: continue
        templ=_resize(patch.astype(np.uint8),width,height).astype(np.float32); mask=_resize(patch_mask.astype(np.uint8),width,height,nearest=True).astype(bool); templ = templ * mask; denom=max(np.sqrt((templ*templ).sum()),1e-6); scores=signal.correlate2d(gray,templ,mode="valid") / denom / np.sqrt(np.maximum(signal.correlate2d(gray*gray,mask.astype(float),mode="valid"),1e-6))
        for _ in range(8):
            flat=int(np.argmax(scores)); loc=(flat % scores.shape[1], flat // scores.shape[1]); score=float(scores[loc[1],loc[0]])
            # NCC is intentionally a high-precision baseline; weaker matches
            # become false positives quickly without a learned descriptor.
            if score<.96: break
            peaks.append((float(score),loc[0],loc[1],width,height)); scores[loc[1]:loc[1]+max(1,height//2),loc[0]:loc[0]+max(1,width//2)] = -1
    out=[]; accepted_centers=[]
    for score,x,y,w,h in sorted(peaks,reverse=True):
        center=np.array([x+w/2.,y+h/2.])
        # Suppress scale-jittered peaks around the same object before SAM sees
        # them; ordinary mask-IoU NMS is too weak when template scale changes.
        if any(np.linalg.norm(center-old_center) < 0.55 * min(w,h,old_size) for old_center,old_size in accepted_centers):
            continue
        mask=np.zeros(exemplar.shape,bool); resized=_resize(exemplar[y0:y1,x0:x1].astype(np.uint8),w,h,nearest=True).astype(bool); mask[y:y+h,x:x+w]=resized[:mask.shape[0]-y,:mask.shape[1]-x]; out.append(InstanceCandidate(mask,score,"template-ncc"))
        accepted_centers.append((center, min(w,h)))
    return out

def _nms(items,max_instances):
    kept=[]
    for item in sorted(items,key=lambda x:x.score,reverse=True):
        if not item.mask.any() or any((item.mask & old.mask).sum()/max((item.mask | old.mask).sum(),1)>.62 for old in kept): continue
        kept.append(item)
        if len(kept)>=max_instances: break
    return kept
