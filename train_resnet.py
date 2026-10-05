import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

torch.manual_seed(42)
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4,0.4,0.4,hue=0.5),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(), norm])
val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), norm])

train_full = datasets.ImageFolder("data", train_tf)
val_full = datasets.ImageFolder("data", val_tf)
classes = train_full.classes
idx = torch.randperm(len(train_full)).tolist()
n = int(0.8 * len(idx))
train_dl = DataLoader(Subset(train_full, idx[:n]), batch_size=16, shuffle=True)
val_dl = DataLoader(Subset(val_full, idx[n:]), batch_size=16)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for ep in range(15):
    model.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    model.eval(); ok = 0
    with torch.no_grad():
        for x, y in val_dl:
            ok += (model(x).argmax(1) == y).sum().item()
    print(f"Epoch {ep+1}: val acc {ok/(len(idx)-n):.2%}")

torch.save({"state": model.state_dict(), "classes": classes}, "model.pth")