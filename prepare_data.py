import os
import csv

dataset_path = r"C:\Users\sneha\Downloads\archive\PlantVillage"

output_file = "labels.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["image", "label"])

    for folder in os.listdir(dataset_path):

        if folder == "PlantVillage":
            continue

        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        for image in os.listdir(folder_path):

            if image.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(folder_path, image)

                writer.writerow([image_path, folder])

print("labels.csv created successfully!")