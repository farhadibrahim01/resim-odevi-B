from pathlib import Path
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

BASE   = Path(__file__).resolve().parent.parent
SRC    = BASE / "data" / "processed"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

for split in SPLITS:
    for cls in SRC.iterdir():
        (BASE / f"data/{split}/{cls.name}").mkdir(parents=True, exist_ok=True)

for cls in SRC.iterdir():
    imgs   = glob(str(cls / "*"))
    train, rest = train_test_split(imgs, test_size=1 - SPLITS["train"], random_state=42, shuffle=True)
    val, test   = train_test_split(rest, test_size=SPLITS["test"] / (SPLITS["val"] + SPLITS["test"]), random_state=42, shuffle=True)

    for split_name, subset in [("train", train), ("val", val), ("test", test)]:
        dst = BASE / f"data/{split_name}/{cls.name}"
        for img in subset:
            shutil.copy(img, dst / Path(img).name)

print("split_data tamamlandı")
