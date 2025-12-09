# import sys
# import os
# import numpy as np
# import pandas as pd
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
# from keras.losses import categorical_crossentropy
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.utils import to_categorical

# df = pd.read_csv("/Users/mayankanand/Documents/Anand_coding/realtime-emotion-detection-keras/Datasets/fer2013.csv")
# # for simplicity, add the dataset in the same folder as your main.py and write:
# # df = pd.read_csv("fer2013.csv")
# # print(df.info())

# X_train, train_y, X_test, test_y = [], [], [], []

# for index, row in df.iterrows():
#     val = row['pixels'].split(" ")
#     try:
#         if 'Training' in row['Usage']:
#             X_train.append(np.array(val, 'float32'))
#             train_y.append(row['emotion'])
#         elif 'PublicTest' in row['Usage']:
#             X_test.append(np.array(val, 'float32'))
#             test_y.append(row['emotion'])
#     except:
#         print(f"Error occurred at index : {index} and row {row}")

# num_features = 64
# num_labels = 7
# batch_size = 64
# epochs = 40
# width, height = 48, 48

# # print(f"X_train sample data : {X_train[0:3]}")
# # print(f"train_y sample data : {train_y[0:3]}")
# # print(f"X_test sample data : {X_test[0:3]}")
# # print(f"test_y sample data : {test_y[0:3]}")

# X_train = np.array(X_train, 'float32')
# train_y = np.array(train_y, 'float32')
# X_test = np.array(X_test, 'float32')
# test_y = np.array(test_y, 'float32')

# train_y = to_categorical(train_y, num_classes=num_labels)
# test_y = to_categorical(test_y, num_classes=num_labels)

# # Normalising the Data between 0 and 1
# # We'll calculate the mean of the data and divide it by the std deviation
# X_train -= np.mean(X_train, axis=0)
# X_train /= np.std(X_train, axis=0)

# X_test -= np.mean(X_test, axis=0)
# X_test /= np.std(X_test, axis=0)

# # print(f"X_train sample data : {X_train[0:3]}")
# # print(f"train_y sample data : {train_y[0:3]}")
# # print(f"X_test sample data : {X_test[0:3]}")
# # print(f"test_y sample data : {test_y[0:3]}")

# X_train = X_train.reshape(X_train.shape[0], width, height, 1)
# X_test = X_test.reshape(X_test.shape[0], width, height, 1)

# # Designing CNN
# model = Sequential()

# # 1st Layer
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
# model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
# model.add(Dropout(0.5))

# # 2nd Convolution Layer
# model.add(Conv2D(num_features, (3, 3), activation='relu'))
# model.add(Conv2D(num_features, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

# # 3rd Convolution Layer
# model.add(Conv2D(num_features*2, (3, 3), activation='relu'))
# model.add(Conv2D(num_features*2, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Flatten())

# # Adding dense layer

# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(num_labels, activation='softmax'))

# model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
# model.fit(X_train, train_y, batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(X_test, test_y),
#           shuffle=True)

# # Saving Model
# fer_json = model.to_json()
# with open("model_fer.json", "w") as json_file:
#     json_file.write(fer_json)
# model.save_weights(".weights.h5")

# main.py — Fixed, MPS-safe EfficientNet-B0 training script
import os
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# -----------------------------
# 0. Basic safety for macOS CPU
# -----------------------------
# Restrict MKL/OpenMP threads to avoid deadlocks / high CPU contention on macs
torch.set_num_threads(1)

# -----------------------------
# 1. Config (change paths here)
# -----------------------------
BATCH_SIZE = 32            # safe default for MPS/CPU; increase if you have lots of RAM
IMG_SIZE = 224
NUM_CLASSES = 7
EPOCHS = 50
LR = 2e-5
TRAIN_PATH = "/Users/mayankanand/Documents/Anand_coding/realtime-emotion-detection-keras/Datasets/archive (6)/train"
TEST_PATH = "/Users/mayankanand/Documents/Anand_coding/realtime-emotion-detection-keras/Datasets/archive (6)/test"
CHECKPOINT_PATH = "best_efficientnet_b0.pth"

# Use MPS if available, otherwise cuda, otherwise cpu
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

# Whether to use autocast (only use it when CUDA is available)
use_amp = (device.type == "cuda")

# -----------------------------
# 2. Force RGB loader helper
# -----------------------------
def pil_loader_rgb(path: str):
    # Always convert to RGB (handles grayscale images)
    return Image.open(path).convert("RGB")

# -----------------------------
# 3. Data augmentation
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# 4. Datasets + loaders
# -----------------------------
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transform)

# Force loader to always give RGB images (fixes grayscale issues)
train_dataset.loader = pil_loader_rgb
test_dataset.loader = pil_loader_rgb

# Balanced sampler to mitigate class imbalance
targets = [label for _, label in train_dataset.samples]
class_counts = Counter(targets)
print("Train class counts:", class_counts)

# weight for each sample = 1 / (#samples in that class)
weights = [1.0 / class_counts[label] for label in targets]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# -----------------------------
# 5. Model (EfficientNet-B0)
# -----------------------------
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Unfreeze most layers (freeze only the first two blocks)
for name, param in model.named_parameters():
    if ("blocks.0" in name) or ("blocks.1" in name):
        param.requires_grad = False
    else:
        param.requires_grad = True

# Replace classifier with a stronger head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

model = model.to(device)

# -----------------------------
# 6. Loss, optimizer, scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -----------------------------
# 7. Training & evaluation
# -----------------------------
best_acc = 0.0

def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(train_loader, desc="Train", leave=False)
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            # Only use autocast on CUDA
            from torch.cuda.amp import autocast, GradScaler
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            # use GradScaler for CUDA
            scaler = GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        iterator.set_postfix(loss=running_loss/total, acc=100.0*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate():
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    iterator = tqdm(test_loader, desc="Eval ", leave=False)
    with torch.no_grad():
        for images, labels in iterator:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            iterator.set_postfix(loss=running_loss/total, acc=100.0*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -----------------------------
# 8. Training loop with checkpointing
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()
    scheduler.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, CHECKPOINT_PATH)
        print(f"  Saved best model (acc: {best_acc*100:.2f}%) -> {CHECKPOINT_PATH}")

print("Training complete. Best val acc: {:.2f}%".format(best_acc*100))
 