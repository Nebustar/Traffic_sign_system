import os
import argparse 
import cv2


try:
    from .config import PATHS,TRAIN
except ImportError:
    from config import PATHS,TRAIN
from ultralytics import YOLO

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")


PREDICT_DEFAULT={
    "weights": PATHS["Best_model"],
    "source":PATHS["Data_yaml"].parent/"images"/"test"/"00006.png",
    "img_size":TRAIN["img_size"],
    "conf":0.65,
    "name":"predict_result",
}

def get_args():
    parser=argparse.ArgumentParser(description="Predict with YOLOv8n model")
    parser.add_argument("--weights",type=str)
    parser.add_argument("--source",type=str)
    parser.add_argument("--img_size",type=int)
    parser.add_argument("--conf",type=float)
    parser.add_argument("--device",type=int,default=TRAIN["device"])
    parser.add_argument("--name",type=str)
    parser.add_argument("--save_txt",action="store_true")

    return parser.parse_args()

def build_predict_config(args):
    weights=args.weights if args.weights is not None else PREDICT_DEFAULT["weights"]

    predict_config={
        "source":args.source if args.source is not None else PREDICT_DEFAULT["source"],
        "imgsz":args.img_size if args.img_size is not None else PREDICT_DEFAULT["img_size"],
        "conf":args.conf if args.conf is not None else PREDICT_DEFAULT["conf"],
        "device":args.device,
        "project":str(PATHS["Runs_dir"]/"predict"),
        "name":args.name if args.name is not None else PREDICT_DEFAULT["name"],
        "save":True,
        "save_txt":args.save_txt,
        "exist_ok":True,
    }

    return weights,predict_config

def print_config(title,config):
    print("\n"+title+"\n")
    for k,v in config.items():
        print(k,":",v)

model=YOLO(str(PATHS["Best_model"]))
def detect_and_annotate(input_path,output_path):

    result=model.predict(
        source=input_path,
        imgsz=TRAIN["img_size"],
        conf=0.65,
        save=False,
        verbose=False,
    )

    img=cv2.imread(input_path)

    detections=[]

    for box in result[0].boxes:
        cls_id=int(box.cls[0])
        confidence=float(box.conf[0])
        xmin,ymin,xmax,ymax=[int(i) for i in box.xyxy[0].tolist()]

        label=result[0].names[cls_id]
        conf_text=f"{confidence*100:.1f}%"

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        cv2.putText(img,f"{label} {conf_text}",(xmin,max(ymin-10,20)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        detections.append((label,conf_text))

    cv2.imwrite(output_path,img)

    if len(detections)==0:
        return "no detection","0.0%"
    
    labels=[d[0] for d in detections]
    confidences=[d[1] for d in detections]

    return ", ".join(labels),", ".join(confidences)



if __name__=="__main__":
    args=get_args()
    weights,predict_config=build_predict_config(args)

    print("\nWeights\n")
    print(weights)

    print_config("Predict config",predict_config)

    print("\nLoad model:\n")
    model=YOLO(str(weights))

    print("\nStart prediction\n")
    model.predict(**predict_config)

    print("\nPredict finished")