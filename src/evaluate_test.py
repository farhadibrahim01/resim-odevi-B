from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import timm
import torch.nn as nn

def main():
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    num_workers= 4
    pin_memory = True

    root      = Path(__file__).resolve().parent.parent
    test_dir  = root / "data" / "test"
    ckpt_path = root / "outputs" / "cnn_best.pt"

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=tf)
    test_ld = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    num_cls = len(test_ds.classes)
    model   = timm.create_model("efficientnet_b0", pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_cls)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_ld:
            x = x.to(device, non_blocking=True)
            preds = model(x).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
