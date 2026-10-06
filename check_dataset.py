import os

dataset_path = r"C:\Users\sneha\Downloads\archive\PlantVillage"

folders = os.listdir(dataset_path)

print("Number of folders:", len(folders))

print("\nFolders:")
for folder in folders:
    print(folder)
