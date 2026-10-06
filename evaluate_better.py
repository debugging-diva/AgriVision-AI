import os
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog

with open("model/better_model.pkl", "rb") as file:
    model = pickle.load(file)

class_features = model["class_features"]

test_path = "data/test"

correct = 0
total = 0
max_images = 300

print("Testing better model...")
print("Testing 300 images only...")
print()

for folder in os.listdir(test_path):

    folder_path = os.path.join(test_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        if total >= max_images:
            break

        image_path = os.path.join(folder_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((32, 32))

            rgb = np.array(image, dtype=np.float32) / 255.0
            color_feature = rgb.flatten()

            gray = image.convert("L")

            texture_feature = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            test_feature = np.concatenate([
                color_feature,
                texture_feature
            ])

            best_distance = float("inf")
            prediction = ""

            for disease in class_features:

                distance = np.mean(
                    (class_features[disease] - test_feature) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    prediction = disease

            if prediction == folder:
                correct += 1

            total += 1

            if total % 50 == 0:
                print("Processed:", total)

        except:
            pass

    if total >= max_images:
        break

accuracy = (correct / total) * 100

print()
print("Total test images:", total)
print("Correct predictions:", correct)
print("Better Model Accuracy:", round(accuracy, 2), "%")