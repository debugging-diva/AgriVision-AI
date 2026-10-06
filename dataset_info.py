import os

dataset_path = r"C:\Users\sneha\Downloads\archive\PlantVillage"

folders = os.listdir(dataset_path)

print("Total categories:", len(folders))
print()

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    images = os.listdir(folder_path)

    print(folder, "=", len(images), "images")