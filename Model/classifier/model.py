import torch

from pathlib import Path
from torch import nn
from torchvision import models,transforms

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def select_device(device):
    device=str(device)

    if device=="cpu":
        return torch.device("cpu")
    
    if device.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda is unavailable")
        return torch.device(f"cuda:{device}")
    
    return torch.device(device)

def build_transform(img_size,train=False):
    operations=[
        transforms.Resize((img_size,img_size)),
    ]

    if train:
        operations.extend([
            transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=8),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
    ])
    return transforms.Compose(operations)

def build_resnet18(num_classes,pretrained=False):
    if pretrained:
        weights=models.ResNet18_Weights.DEFAULT
    else:
        weights=None

    model=models.resnet18(weights=weights)

    input_features=model.fc.in_features
    model.fc=nn.Linear(input_features,num_classes)

    return model

def load_checkpoint(checkpoint_path,device):
    checkpoint_path=Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found:{checkpoint_path}")
    
    torch_device=select_device(device)

    checkpoint=torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys={
        "arch",
        "imgsz",
        "class_names",
        "model",
    }
    missing_keys=required_keys-checkpoint.keys()

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing keys:{sorted(missing_keys)}"
        )
    if checkpoint["arch"]!="resnet18":
        raise ValueError(
            f"Expected resnet18,got:{checkpoint['arch']}"
        )
    
    class_names=checkpoint["class_names"]

    model=build_resnet18(
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(torch_device)
    model.eval()

    return model,checkpoint,torch_device

def model_smoke_test():
    model=build_resnet18(
        num_classes=46,
        pretrained=False,
    )
    test_input=torch.randn(2,3,224,224)
    test_output=model(test_input)

    print("input_shape:",tuple(test_input.shape))
    print("output_shape",tuple(test_output.shape))

    assert tuple(test_output.shape)==(2,46)

    print("model smoke test passed")
            
if __name__=="__main__":
    model_smoke_test()
