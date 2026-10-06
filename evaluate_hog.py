import os
import pickle
from PIL import Image
from skimage.feature import hog

with open("model/hog_model.pkl", "rb") as file:
    model = pickle.load(file)

features = model["features"]
labels = model["labels"]

test_path = "data/test"

correct = 0
total = 0

print("Testing HOG model...")

for folder in os.listdir(test_path):

    folder_path = os.path.join(test_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print("Testing:", folder)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        try:
            image = Image.open(image_path).convert("L")
            image = image.resize((64, 64))

            test_feature = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            best_distance = float("inf")
            prediction = ""

            for i in range(len(features)):

                distance = sum(
                    (features[i][j] - test_feature[j]) ** 2
                    for j in range(len(test_feature))
                )

                if distance < best_distance:
                    best_distance = distance
                    prediction = labels[i]

            if prediction == folder:
                correct += 1

            total += 1

        except:
            pass

accuracy = (correct / total) * 100

print()
print("Total test images:", total)
print("Correct predictions:", correct)
print("HOG Accuracy:", round(accuracy, 2), "%")