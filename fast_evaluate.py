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

print("Fast HOG evaluation started...")
print("Testing 300 images only...\n")

count = 0
max_images = 300

for folder in os.listdir(test_path):

    folder_path = os.path.join(test_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        if count >= max_images:
            break

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

            # Check only 1000 training images
            best_distance = float("inf")
            prediction = ""

            step = max(1, len(features) // 1000)

            for i in range(0, len(features), step):

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
            count += 1

            if count % 50 == 0:
                print("Processed:", count)

        except:
            pass

    if count >= max_images:
        break

accuracy = (correct / total) * 100

print()
print("Total test images:", total)
print("Correct predictions:", correct)
print("Fast HOG Accuracy:", round(accuracy, 2), "%")