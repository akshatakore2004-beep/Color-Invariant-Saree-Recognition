import cv2, sys, joblib
sys.path.insert(0, "src")
from features import extract
model = joblib.load("model.pkl")
img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for shift in [0, 30, 60, 90, 120, 150]:
    h = hsv.copy()
    h[..., 0] = (h[..., 0].astype(int) + shift) % 180
    cv2.imwrite("tmp.jpg", cv2.cvtColor(h, cv2.COLOR_HSV2BGR))
    print("hue shift", shift, "->", model.predict([extract("tmp.jpg")])[0])
