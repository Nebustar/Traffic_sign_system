import os
import torch
import ultralytics

from config import PATHS

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")


ENV_INFO={
    "torch_version":torch.__version__,
    "cuda_available":torch.cuda.is_available(),
    "cuda_device_count":torch.cuda.device_count(),
    "ultralytics_version":ultralytics.__version__,
}

if __name__=="__main__":
    print("\nENV_INFO:\n")
    for key,value in ENV_INFO.items():
        print(key,":",value)