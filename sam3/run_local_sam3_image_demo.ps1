$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "D:\SAM\sam3\.deps_runtime;D:\SAM\sam3"

D:\SAM\conda_envs\sam_gpu\python.exe -B `
  D:\SAM\sam3\examples\local_sam3_image_demo.py `
  --image D:\SAM\sam3\assets\images\test_image.jpg `
  --prompt child `
  --checkpoint D:\SAM\assets\checkpoints\sam3_modelscope\sam3.pt `
  --out-dir D:\SAM\sam3\outputs
