import pickle
from PIL import Image

with open("model/plant_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

features = model["features"]
labels = model["labels"]

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = image.resize((32, 32))

pixels = list(image.getdata())

red = sum(p[0] for p in pixels) / len(pixels)
green = sum(p[1] for p in pixels) / len(pixels)
blue = sum(p[2] for p in pixels) / len(pixels)

test_feature = [red, green, blue]

best_distance = float("inf")
best_label = ""

for i in range(len(features)):

    r = features[i][0] - test_feature[0]
    g = features[i][1] - test_feature[1]
    b = features[i][2] - test_feature[2]

    distance = (r*r + g*g + b*b) ** 0.5

    if distance < best_distance:
        best_distance = distance
        best_label = labels[i]

print()
print("🌱 Predicted Disease:", best_label)