import os
from torchvision import datasets, models, transforms
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# ===================================
# SETTINGS
# ===================================
DATA_DIR = "dataset"  # path to your dataset
MODEL_PATH = "best_garbage_model.pt"
NUM_CLASSES = 6  # adjust if your dataset has fewer or more
BATCH_SIZE = 16
EPOCHS = 5  # try small number first for testing
LEARNING_RATE = 0.001
TRAIN_VAL_SPLIT = 0.8  # if dataset not already split

# ===================================
# DEVICE SETUP
# ===================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================================
# DATA TRANSFORMS
# ===================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===================================
# LOAD DATA
# ===================================
if os.path.exists(os.path.join(DATA_DIR, "train")):
    print("Using existing train/val split...")
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)
else:
    print("No split found. Splitting automatically...")
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    total_len = len(full_dataset)
    train_len = int(total_len * TRAIN_VAL_SPLIT)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = train_dataset.dataset.classes if hasattr(train_dataset, 'dataset') else train_dataset.classes
print("Classes:", CLASS_NAMES)

# ===================================
# MODEL SETUP
# ===================================
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===================================
# TRAINING LOOP
# ===================================
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Validation accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Saved new best model!")

print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
