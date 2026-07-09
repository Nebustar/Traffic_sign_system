import os
import argparse 

from config import PATHS,TRAIN
from ultralytics import YOLO

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")

Smoke_default={
    "epochs":1,
    "img_size":320,
    "batch_size":4,
    "fraction":0.01,
    "name":"smoke_train",
}

Full_default={
    "epochs":TRAIN["epochs"],
    "img_size":TRAIN["img_size"],
    "batch_size":TRAIN["batch_size"],
    "fraction":1.0,
    "name":"full_train",
}

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n")

    parser.add_argument("--mode", type=str, choices=["smoke", "full"], default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--name", type=str)

    parser.add_argument("--device", type=int, default=TRAIN["device"])
    parser.add_argument("--workers", type=int, default=TRAIN["workers"])

    return parser.parse_args()

def build_train_config(args):
    if args.mode=="smoke":
        default_config=Smoke_default
    else:
        default_config=Full_default

    train_config={
        "data":str(PATHS["Data_yaml"]),
        "epochs":args.epochs if args.epochs is not None else default_config["epochs"],
        "imgsz":args.img_size if args.img_size is not None else default_config["img_size"],
        "batch":args.batch_size if args.batch_size is not None else default_config["batch_size"],
        "fraction":args.fraction if args.fraction is not None else default_config["fraction"],
        "name":args.name if args.name is not None else default_config["name"],
        "device":args.device,
        "workers":args.workers,
        "project":str(PATHS["Runs_dir"]/"detect"),
        "exist_ok":True,
    }

    return train_config

def print_config(config):
    print("\nTrain Config\n")

    for k,v in config.items():
        print(k,":",v)

if __name__=="__main__":
    args=get_args()
    train_config=build_train_config(args)

    print_config(train_config)

    print("\n-------------Load model-------------\n")
    model=YOLO(str(PATHS["Pretrained_model"]))

    print("\n-------------Start training-------------\n")
    model.train(**train_config)

    print("\n-------------Train finished-------------\n")
