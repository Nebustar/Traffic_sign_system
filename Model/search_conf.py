import os
import argparse
import csv

from config import PATHS,TRAIN
from ultralytics import YOLO

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")


Search_default={
    "weights":PATHS["Best_model"],
    "split":"val",
    "start":0.30,
    "end":0.80,
    "step":0.05,
    "img_size":TRAIN["img_size"],
    "batch_size":TRAIN["batch_size"],
    "name":"conf_search_val",
}


def get_args():
    parser=argparse.ArgumentParser(description="Search best confidence threshold")

    parser.add_argument("--weights",type=str)
    parser.add_argument("--split",type=str,choices=["train","val","test"])
    parser.add_argument("--start",type=float)
    parser.add_argument("--end",type=float)
    parser.add_argument("--step",type=float)
    parser.add_argument("--img_size",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--device",type=int,default=TRAIN["device"])
    parser.add_argument("--workers",type=int,default=TRAIN["workers"])
    parser.add_argument("--name",type=str)

    return parser.parse_args()


def build_search_config(args):
    weights=args.weights if args.weights is not None else Search_default["weights"]

    search_config={
        "split":args.split if args.split is not None else Search_default["split"],
        "start":args.start if args.start is not None else Search_default["start"],
        "end":args.end if args.end is not None else Search_default["end"],
        "step":args.step if args.step is not None else Search_default["step"],
        "imgsz":args.img_size if args.img_size is not None else Search_default["img_size"],
        "batch":args.batch_size if args.batch_size is not None else Search_default["batch_size"],
        "device":args.device,
        "workers":args.workers,
        "name":args.name if args.name is not None else Search_default["name"],
    }

    return weights,search_config


def make_conf_list(start,end,step):
    conf_list=[]
    current=start

    while current<=end+1e-9:
        conf_list.append(round(current,2))
        current+=step

    return conf_list


def get_box_metrics(metrics):
    precision=float(metrics.box.mp)
    recall=float(metrics.box.mr)
    map50=float(metrics.box.map50)
    map50_95=float(metrics.box.map)
    f1=0.0

    if precision+recall>0:
        f1=2*precision*recall/(precision+recall)

    return {
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "map50":map50,
        "map50_95":map50_95,
    }


def print_config(config):
    print("\nSearch Config\n")

    for k,v in config.items():
        print(k,":",v)


def print_row(row):
    print(
        f"conf={row['conf']:.2f} "
        f"precision={row['precision']:.4f} "
        f"recall={row['recall']:.4f} "
        f"f1={row['f1']:.4f} "
        f"mAP50={row['map50']:.4f} "
        f"mAP50-95={row['map50_95']:.4f}"
    )


if __name__=="__main__":
    args=get_args()
    weights,search_config=build_search_config(args)

    save_dir=PATHS["Runs_dir"]/"search_conf"/search_config["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path=save_dir/"search_conf_results.csv"

    print("\nWeights\n")
    print(weights)

    print_config(search_config)

    print("\n-------------Load model-------------\n")
    model=YOLO(str(weights))

    conf_list=make_conf_list(search_config["start"],search_config["end"],search_config["step"])

    rows=[]

    print("\n-------------Start searching confidence threshold-------------\n")
    for conf in conf_list:
        metrics=model.val(
            data=str(PATHS["Data_yaml"]),
            split=search_config["split"],
            imgsz=search_config["imgsz"],
            batch=search_config["batch"],
            conf=conf,
            device=search_config["device"],
            workers=search_config["workers"],
            project=str(PATHS["Runs_dir"]/"search_conf"),
            name=search_config["name"],
            plots=False,
            verbose=False,
            exist_ok=True,
        )

        row={"conf":conf}
        row.update(get_box_metrics(metrics))
        rows.append(row)
        print_row(row)

    best_row=max(rows,key=lambda item:item["f1"])

    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["conf","precision","recall","f1","map50","map50_95"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n-------------Best confidence by F1-------------\n")
    print_row(best_row)

    print("\n-------------Saved results-------------\n")
    print(csv_path)

    print("\n-------------Search finished-------------\n")
