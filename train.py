import os, glob, joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from features import extract

X, y = [], []
for cls in os.listdir("data"):
    for f in glob.glob(f"data/{cls}/*"):
        v = extract(f)
        if v is not None:
            X.append(v); y.append(cls)

print("Total images:", len(X))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = SVC(kernel="rbf", C=10, probability=True).fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte)))
joblib.dump(model, "model.pkl")
