import os
import argparse 
import cv2

import torch
from pathlib import Path
from PIL import Image

try:
    from .config import PATHS,TRAIN,CLASSIFIER,TWO_STAGE
    from .classifier.model import build_transform,load_checkpoint
except ImportError:
    from config import PATHS,TRAIN,CLASSIFIER,TWO_STAGE
    from classifier.model import build_transform,load_checkpoint
from ultralytics import YOLO

os.environ["YOLO_CONFIG_DIR"]=str(PATHS["Runs_dir"].parent/".cache"/"yolo")


PREDICT_DEFAULT={
    "weights": PATHS["Detector_model"],
    "source":PATHS["Data_yaml"].parent/"images"/"test"/"10056.jpg",
    "img_size":TWO_STAGE["detector_img_size"],
    "conf":TWO_STAGE["detector_conf"],
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

detector=YOLO(str(PATHS["Detector_model"]))
classifier,classifier_checkpoint,torch_device=load_checkpoint(
    PATHS["Classifier_model"],
    TWO_STAGE["device"],
)

class_names=classifier_checkpoint["class_names"]
classifier_transform=build_transform(
    classifier_checkpoint["imgsz"],
    train=False,
)

'''
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
'''

def detect_and_annotate(
    input_path,
    output_path,
    detector_model=None,
    img_size=None,
    detector_conf=None,
    device=None,
):
    if detector_model is None:
        detector_model=detector

    if img_size is None:
        img_size=TWO_STAGE["detector_img_size"]

    if detector_conf is None:
        detector_conf=TWO_STAGE["detector_conf"]

    if device is None:
        device=TWO_STAGE["device"]

    image=cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    results=detector_model.predict(
        source=str(input_path),
        imgsz=img_size,
        conf=detector_conf,
        device=device,
        save=False,
        verbose=False,
    )

    detections=[]

    for box in results[0].boxes:
        det_confidence=float(box.conf[0])
        xmin,ymin,xmax,ymax=[
            int(value)
            for value in box.xyxy[0].tolist()
        ]

        label,cls_confidence=classify_crop(
            image,
            xmin,
            ymin,
            xmax,
            ymax,
        )

        det_text=f"{det_confidence*100:.1f}%"
        cls_text=f"{cls_confidence*100:.1f}%"
        display_text=f"{label} det:{det_text} cls:{cls_text}"

        cv2.rectangle(
            image,
            (xmin,ymin),
            (xmax,ymax),
            (0,255,0),
            2,
        )

        cv2.putText(
            image,
            display_text,
            (xmin,max(ymin-10,20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2,
        )

        detections.append((
            label,
            f"det:{det_text} cls:{cls_text}",
        ))

    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    if not cv2.imwrite(str(output_path),image):
        raise RuntimeError(f"Cannot save image: {output_path}")

    if len(detections)==0:
        return "no detection","0.0%"

    labels=[item[0] for item in detections]
    confidences=[item[1] for item in detections]

    return ", ".join(labels),", ".join(confidences)


def classify_crop(image,xmin,ymin,xmax,ymax):
    image_height,image_width=image.shape[:2]

    box_width=xmax-xmin
    box_height=ymax-ymin

    padding_x=int(box_width*CLASSIFIER["padding"])
    padding_y=int(box_height*CLASSIFIER["padding"])

    crop_x1=max(0,xmin-padding_x)
    crop_y1=max(0,ymin-padding_y)
    crop_x2=min(image_width,xmax+padding_x)
    crop_y2=min(image_height,ymax+padding_y)

    if crop_x2<=crop_x1 or crop_y2<=crop_y1:
        raise ValueError("Invalid crop box")
    
    crop=image[crop_y1:crop_y2,crop_x1:crop_x2]
    crop_rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    crop_image=Image.fromarray(crop_rgb)

    tensor=classifier_transform(crop_image)
    tensor=tensor.unsqueeze(0).to(torch_device)

    with torch.inference_mode():
        logits=classifier(tensor)
        probabilities=torch.softmax(logits,dim=1)[0]

    cls_confidence,class_id=probabilities.max(dim=0)

    label=class_names[int(class_id.item())]
    return label,float(cls_confidence.item())
    
'''
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
'''

if __name__=="__main__":
    args=get_args()
    weights,predict_config=build_predict_config(args)

    print("\nWeights\n")
    print(weights)

    print_config("Predict config",predict_config)

    print("\nLoad detector:\n")
    cli_detector=YOLO(str(weights))

    source_path=Path(predict_config["source"])
    output_dir=(
        Path(predict_config["project"])
        /predict_config["name"]
    )
    output_path=output_dir/source_path.name

    print("\nStart two-stage prediction\n")

    labels,confidences=detect_and_annotate(
        input_path=source_path,
        output_path=output_path,
        detector_model=cli_detector,
        img_size=predict_config["imgsz"],
        detector_conf=predict_config["conf"],
        device=predict_config["device"],
    )

    print("labels:",labels)
    print("confidences:",confidences)
    print("output:",output_path)
    print("\nPredict finished")