import os, glob, joblib, cv2, numpy as np, sys
sys.path.insert(0, "src")
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import extract

def shifted(path, shift):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 180
    cv2.imwrite("tmp.jpg", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    return extract("tmp.jpg")

files, labels = [], []
for cls in os.listdir("data"):
    for f in glob.glob(f"data/{cls}/*"):
        if cv2.imread(f) is not None:
            files.append(f); labels.append(cls)

ftr, fte, ytr, yte = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

Xtr, Ytr = [], []
for f, l in zip(ftr, ytr):
    for s in [0, 45, 90, 135]:
        Xtr.append(shifted(f, s)); Ytr.append(l)

Xte, Yte = [], []
for f, l in zip(fte, yte):
    for s in [0, 60, 120]:
        Xte.append(shifted(f, s)); Yte.append(l)

model = SVC(kernel="rbf", C=10).fit(Xtr, Ytr)
print(classification_report(Yte, model.predict(Xte)))
joblib.dump(model, "model.pkl")
