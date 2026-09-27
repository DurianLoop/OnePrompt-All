# Local compatibility changes

This directory is a source snapshot of FoB (upstream commit `40d0e23d75f81f07bb7f34cb641db51ef490d025`) copied for a protocol-matched MoNuSeg experiment.

Only two execution repairs were made:

1. `models/encoder.py` reads the official ResNet-101 checkpoint location from `FOB_RESNET101_COCO` rather than a literal placeholder.
2. `info_nce.py` supplies the missing dependency required by the release import path. The released FoB forward constructs, but does not invoke, this loss.
3. The legacy list-based advanced index in `models/FoB.py` is expressed as an equivalent slice (`0:1`) so that PyTorch 2.11 retains the release's intended single-item batch shape instead of relying on deprecated indexing semantics.
4. The upstream evaluation guard no longer checks `query_labels.max()`. Query labels are used for the released training losses only; removing that redundant inference-time guard makes held-out FoB prompt generation strictly GT-free.

No architecture, prompt-generation, loss, optimizer, evaluation metric, SAM backbone, or split policy has been changed. Further changes require a separate recorded patch and invalidate comparisons to the upstream method until audited.
