import os
import argparse

from config import PATHS,TRAIN
from ultralytics import YOLO

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")

EVAL_DEFAULT={
    "weights": PATHS["Best_model"],
    "split":"test",
    "img_size":TRAIN["img_size"],
    "batch_size":TRAIN["batch_size"],
    "name":"eval_result",
}

def get_args():
    parser=argparse.ArgumentParser(description="Evaluate YOLOv8n model")
    parser.add_argument("--weights",type=str)
    parser.add_argument("--split",type=str,choices=["val","test"])
    parser.add_argument("--img_size",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--device",type=int,default=TRAIN["device"])
    parser.add_argument("--workers",type=int,default=TRAIN["workers"])
    parser.add_argument("--name",type=str)

    return parser.parse_args()

def build_eval_config(args):
    weights=args.weights if args.weights is not None else EVAL_DEFAULT["weights"]

    eval_config={
        "data":str(PATHS['Data_yaml']),
        "split":args.split if args.split is not None else EVAL_DEFAULT["split"],
        "imgsz":args.img_size if args.img_size is not None else EVAL_DEFAULT["img_size"],
        "batch":args.batch_size if args.batch_size is not None else EVAL_DEFAULT["batch_size"],
        "device":args.device,
        "workers":args.workers,
        "project":str(PATHS["Runs_dir"]/"eval"),
        "name":args.name if args.name is not None else EVAL_DEFAULT["name"],
        "exist_ok":True,
    }

    return weights,eval_config

def print_config(title,config):
    print("\n"+title+"\n")
    for k,v in config.items():
        print(k,":",v)

if __name__=="__main__":
    args=get_args()
    weights,eval_config=build_eval_config(args)

    print("\nWeights\n")
    print(weights)

    print_config("Eval config",eval_config)

    print("\nLoad model:\n")
    model=YOLO(str(weights))

    print("\nStart evaluation\n")
    model.val(**eval_config)

    print("\nEvaluation finished")