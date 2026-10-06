import os
import pickle
import numpy as np
from PIL import Image
from skimage.feature import hog

train_path = "data/train"

class_features = {}
class_counts = {}

print("Training better plant disease model...")
print()

for folder in os.listdir(train_path):

    folder_path = os.path.join(train_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", folder)

    all_features = []

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((32, 32))

            # Color features
            rgb = np.array(image, dtype=np.float32) / 255.0
            color_feature = rgb.flatten()

            # Texture/shape features
            gray = image.convert("L")

            texture_feature = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2)
            )

            # Combine color + texture
            feature = np.concatenate([
                color_feature,
                texture_feature
            ])

            all_features.append(feature)

        except:
            pass

    if len(all_features) > 0:

        # Create one representative feature for this disease
        class_features[folder] = np.mean(
            np.array(all_features),
            axis=0
        )

        class_counts[folder] = len(all_features)

print()
print("Creating model...")

model = {
    "class_features": class_features,
    "class_counts": class_counts
}

os.makedirs("model", exist_ok=True)

with open("model/better_model.pkl", "wb") as file:
    pickle.dump(model, file)

print()
print("BETTER MODEL CREATED!")
print("Classes:", len(class_features))
print("Model saved as:")
print("model/better_model.pkl")