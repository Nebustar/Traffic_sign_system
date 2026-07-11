from pathlib import Path

Project_root=Path(__file__).resolve().parent
Repo_root=Project_root.parent

DATASETS={
    "hust":Repo_root/"data"/"hust_yolo"/"data.yaml",
    "tt100k":Repo_root/"data"/"tt100k_yolo"/"data.yaml",
}

DEFAULT_DATASET="tt100k"

PATHS={
    "Data_yaml":DATASETS[DEFAULT_DATASET],
    "Pretrained_model":Project_root/"weights"/"yolov8n.pt",

    "Detector_model":Project_root/"weights"/"tt100k_detector_best.pt",
    "Classifier_model":Project_root/"weights"/"tt100k_resnet18_best.pt",

    "Best_model":Project_root/"weights"/"tt100k_detector_best.pt",
    "Class_file":Project_root/"configs"/"classes.txt",
    "Runs_dir":Project_root/"runs",
}

CLASSIFIER={
    "arch":"resnet18",
    "num_classes":46,
    "img_size":224,
    "padding":0.12,
}

TWO_STAGE={
    "detector_img_size":1280,
    "detector_conf":0.25,
    "device":"auto",
}

TRAIN={
    "epochs":50,
    "img_size":960,
    "batch_size":16,
    "device":0,
    "workers":0,
    "fraction":1.0,
    "patience":20,
    "amp":True,
    "cache":False,
    "single_cls":False,
    "close_mosaic":10,
}

if __name__=="__main__":
    print("project_root",Project_root)

    print("\nPATHS:\n")
    for key,value in PATHS.items():
        print(key,":",value)
        if(value.exists()):
            print(key," exists=true")
        else:
            print(key," exists=false")

    print("\nTRAIN:\n")
    for key,value in TRAIN.items():
        print(key,":",value)
