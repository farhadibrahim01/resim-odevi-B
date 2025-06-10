from pathlib import Path
import torch
import timm
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def main():
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    batch_size  = 128
    epochs      = 15
    patience    = 3
    pin_memory  = True
    num_workers = 4

    root      = Path(__file__).resolve().parent.parent
    data_dir  = root / "data" / "raw"
    ckpt_path = root / "outputs" / "cnn_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir, transform=val_tf)

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

    num_cls = len(train_ds.classes)

    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_cls)
    for p in model.parameters(): p.requires_grad = False
    for p in model.blocks[-2:].parameters(): p.requires_grad = True
    for p in model.classifier.parameters(): p.requires_grad = True
    model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler  = GradScaler()

    best_acc   = 0.0
    no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_ld, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                preds = model(x)
                loss  = loss_fn(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad(), autocast(device_type="cuda", enabled=(device=="cuda")):
            for x, y in val_ld:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total   += y.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc    = 100 * correct / total
        print(f"Epoch {epoch}/{epochs} | loss {train_loss:.4f} | val_acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    print(f"Best validation accuracy: {best_acc:.2f}% — saved to {ckpt_path.name}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
