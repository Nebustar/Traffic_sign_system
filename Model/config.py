from pathlib import Path

Project_root=Path(__file__).resolve().parent

PATHS={
    "Data_yaml":Project_root/"data"/"hust_yolo"/"data.yaml",
    "Pretrained_model":Project_root/"weights"/"yolov8n.pt",
    "Best_model":Project_root/"weights"/"best.pt",
    "Class_file":Project_root/"configs"/"classes.txt",
    "Runs_dir":Project_root/"runs",
}

TRAIN={
    "epochs":50,
    "img_size":960,
    "batch_size":16,
    "device":0,
    "workers":2,
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
    