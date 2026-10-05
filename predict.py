import sys, joblib
from features import extract

model = joblib.load("model.pkl")
print("Design:", model.predict([extract(sys.argv[1])])[0])
