# OnePrompt-All

**One rough prompt, all same-class instances in one image.**

OnePrompt-All is a training-free framework and interactive workbench for
**single-image, single-exemplar, multi-instance segmentation**. A user clicks
one object, draws a loose box, or circles an approximate region. The system
then infers the intended visual concept inside that same image and returns a
separate mask for every matching instance—without a class name, an external
reference image, a reference mask, or an object count.

![SAM Prompt Benchmark workbench](docs/assets/fig3_workbench.jpg)

The workbench also retains comparable SAM-series and Matcher-series modes, so
prompted segmentation, cross-image reference matching, and OnePrompt-All can
be inspected in one browser interface.

## Workbench

The web application is `surf/demos/medical_sam_click_app.py`. In the
**OnePrompt-All** mode, choose an image, then select one of the three prompt
types at the top right of the canvas:

1. **Click** — click anywhere inside one target instance.
2. **Box** — draw a loose rectangle; background may be included.
3. **Circle** — drag a rough circular region; background may be included.

One user gesture triggers one complete inference. The output panel shows every
accepted instance as an individual colored mask. The user does not need to
provide iterative positive/negative clicks.

### Environment

The tested local configuration is Windows, Python 3.12, CUDA 12.6+ and a
CUDA-capable GPU. SAM3 itself follows Meta's current Python/PyTorch
requirements; see the official [SAM3 installation guide](https://github.com/facebookresearch/sam3#installation).

```powershell
# 1) Create and activate an environment
conda create -n oneprompt-all python=3.12 -y
conda activate oneprompt-all

# 2) Install a CUDA build of PyTorch appropriate for your machine.
# Example from the SAM3 guide (adjust if your CUDA/PyTorch version differs):
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) Install the workbench dependencies
pip install -r requirements-demo.txt

# 4) Install SAM3 beside this repository
git clone https://github.com/facebookresearch/sam3.git external/sam3
pip install -e external/sam3
```

`requirements-demo.txt` contains the lightweight web/UI dependencies. If a
version conflict occurs, keep the PyTorch version required by SAM3 and install
the remaining packages without forcing the legacy `torch` pins in
`surf/requirements.txt`.

### Model weights

Weights are intentionally excluded from Git. Create the following directory:

```text
assets/
└── checkpoints/
    ├── sam3_modelscope/
    │   └── sam3.pt
    └── dinov2_vitb14_pretrain.pth
```

Required files for OnePrompt-All:

| Component | File expected by the workbench | Obtain it from |
|---|---|---|
| SAM3 instance generator | `assets/checkpoints/sam3_modelscope/sam3.pt` | The official [SAM3 Hugging Face repository](https://huggingface.co/facebook/sam3). Request access, run `hf auth login`, and download a compatible image checkpoint. If its filename differs, either rename it to `sam3.pt` or pass `--sam3-checkpoint <path>`. |
| DINOv2-B/14 feature encoder | `assets/checkpoints/dinov2_vitb14_pretrain.pth` | The official [DINOv2 repository](https://github.com/facebookresearch/dinov2#pretrained-models). Download the ViT-B/14 backbone checkpoint and save it using this filename. |

The app checks these locations at startup/on demand and reports a clear error
when a required file is absent. Other SAM, SAM2, MedSAM, and Matcher checkpoints
are optional: they enable the corresponding workbench modes but are not needed
for OnePrompt-All.

### Run

From the repository root:

```powershell
python .\surf\demos\medical_sam_click_app.py --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860>. Do not start a second server if one is
already listening on port 7860.

Optional explicit paths are useful if weights live elsewhere:

```powershell
python .\surf\demos\medical_sam_click_app.py `
  --host 127.0.0.1 --port 7860 `
  --sam3-checkpoint D:\models\sam3.pt
```

### Included example data

`surf/datasets/oneprompt_all/natural20/` contains 20 small natural-image
examples for browsing in the workbench. Full training/evaluation datasets,
medical datasets, and all checkpoints remain local and are ignored by Git.

## OnePrompt-All method

![OnePrompt-All architecture](docs/assets/fig1_architecture2.jpg)

Given a single coarse prompt \(p\) in image \(I\), the method constructs an
online exemplar rather than relying on an external reference:

1. **Prompt parsing and seed resolution.** SAM3 converts the point, loose box,
   or circle into a target seed mask.
2. **Feature purification.** DINOv2-B/14 tokens pooled inside the seed form an
   image-specific exemplar representation.
3. **Same-image candidate discovery.** SAM3 produces instance candidates across
   the image; DINO similarity is used as *soft evidence*, not as a hard veto.
4. **Candidate-set selection.** SAM confidence, soft feature consistency,
   minimum-area filtering, and mask NMS suppress duplicates and fragments while
   retaining visually varying instances.
5. **Batch instance output.** The retained masks are returned separately, which
   supports segmentation and counting in the same interaction.

Dense scenes are handled without fixed automatic negative points by default:
an apparent background patch can actually be another true target instance.

### Evaluation snapshot

On the integrated benchmark of 2,220 same-image multi-instance images from
232 categories, evaluated with five random interior clicks per image:

| Method | Input at inference | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| SAM3 Text Oracle | Correct class name | 0.691 | 0.737 | 0.693 |
| OnePrompt-All | One positive point, no class name | 0.667 | 0.687 | 0.712 |

The text-oracle comparison intentionally gives SAM3 the ground-truth class
name. OnePrompt-All receives only the click, yet yields a comparable F1 and
higher recall. See [the method note](surf/research/ONEPROMPT_ALL_METHOD.md) and
[evaluation scripts](surf/research/) for protocol details.

## Poster

The complete research poster is available here:

[Download the OnePrompt-All poster (PDF)](docs/assets/surf_poster.pdf)

## Repository layout

```text
surf/
├── demos/       # FastAPI workbench entrypoint
├── workbench/   # OnePrompt-All and dataset-browser implementation
├── research/    # benchmark construction, evaluation, and analysis scripts
├── adapters/    # SAM/SAM2/SAM3/Matcher adapters
└── datasets/oneprompt_all/natural20/  # small tracked examples
docs/assets/     # workbench, architecture, and poster assets
assets/checkpoints/  # local-only model weights (not tracked)
external/sam3/   # local SAM3 clone (not tracked)
```

## Acknowledgements

OnePrompt-All builds on [SAM3](https://github.com/facebookresearch/sam3),
[DINOv2](https://github.com/facebookresearch/dinov2),
[Segment Anything](https://github.com/facebookresearch/segment-anything), and
[Matcher](https://github.com/aim-uofa/Matcher). Please follow the respective
licenses and checkpoint access terms.
