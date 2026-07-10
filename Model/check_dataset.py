from config import PATHS

Dataset_root=PATHS["Data_yaml"].parent

SPLITS={
    "train":{
        "images":Dataset_root/"images"/"train",
        "labels":Dataset_root/"labels"/"train",
    },
    "val":{
        "images":Dataset_root/"images"/"val",
        "labels":Dataset_root/"labels"/"val",
    },
    "test":{
        "images":Dataset_root/"images"/"test",
        "labels":Dataset_root/"labels"/"test",
    }
}

Image_suffixes=[".png"]

def get_image_files(image_dir):
    image_files=[]

    for file in image_dir.iterdir():
        if file.suffix.lower() in Image_suffixes:
            image_files.append(file)

    return image_files

if __name__=="__main__":
    print("\nDataset_root:",Dataset_root)
    
    print("\nCheck splits:\n")

    for name,paths in SPLITS.items():
        image_dir=paths["images"]
        label_dir=paths["labels"]

        image_files=get_image_files(image_dir)
        label_files=list(label_dir.glob("*.txt"))

        missing_labels=[]

        for image_file in image_files:
            label_file=label_dir/(image_file.stem+".txt")
            if not label_file.exists():
                missing_labels.append(image_file.name)

        CHECK={
            "split":name,
            "image_dir_exists":image_dir.exists(),
            "label_dir_exists":label_dir.exists(),
            "image_count":len(image_files),
            "label_count":len(label_files),
            "missing_label_count":len(missing_labels),
        }

        for k,v in CHECK.items():
            print(k,":",v)
        print("-"*50)

