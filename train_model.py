import os
import pickle
from PIL import Image
from skimage.feature import hog

train_path = "data/train"

features = []
labels = []

print("Reading training images...")

for folder in os.listdir(train_path):

    folder_path = os.path.join(train_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", folder)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        try:
            image = Image.open(image_path).convert("L")
            image = image.resize((64, 64))

            feature = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            features.append(feature)
            labels.append(folder)

        except:
            print("Skipped:", image_name)

print()
print("Images processed:", len(features))

model = {
    "features": features,
    "labels": labels
}

os.makedirs("model", exist_ok=True)

with open("model/hog_model.pkl", "wb") as file:
    pickle.dump(model, file)

print()
print("HOG MODEL CREATED!")
print("Saved as: model/hog_model.pkl")