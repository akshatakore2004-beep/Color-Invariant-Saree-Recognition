import sys, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models

ck = torch.load("model.pth")
classes = ck["classes"]
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ck["state"])
model.eval()

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

img = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    p = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
for c, v in zip(classes, p):
    print(f"{c}: {v:.1%}")
print("Prediction:", classes[p.argmax()])