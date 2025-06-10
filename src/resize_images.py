from pathlib import Path
import cv2
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
OUT  = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = (224, 224)

for cls_dir in RAW.iterdir():
    dest = OUT / cls_dir.name
    dest.mkdir(exist_ok=True)
    for img_path in tqdm(list(cls_dir.glob("*")), desc=cls_dir.name):
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / img_path.name), img)
print("resize_images → tamamlandı")
