import os
import shutil
import random

dataset_path = r"C:\Users\sneha\Downloads\archive\PlantVillage"

output_path = "data"

train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for folder in os.listdir(dataset_path):

    if folder == "PlantVillage":
        continue

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    images = []

    for image in os.listdir(folder_path):

        if image.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(image)

    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_images = images[:split]
    test_images = images[split:]

    os.makedirs(os.path.join(train_path, folder), exist_ok=True)
    os.makedirs(os.path.join(test_path, folder), exist_ok=True)

    for image in train_images:
        shutil.copy(
            os.path.join(folder_path, image),
            os.path.join(train_path, folder, image)
        )

    for image in test_images:
        shutil.copy(
            os.path.join(folder_path, image),
            os.path.join(test_path, folder, image)
        )

print("Dataset split completed!")
print("80% training")
print("20% testing")