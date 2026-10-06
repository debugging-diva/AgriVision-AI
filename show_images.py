import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = r"C:\Users\sneha\Downloads\archive\PlantVillage"

folders = os.listdir(dataset_path)

folder = folders[0]

folder_path = os.path.join(dataset_path, folder)

images = os.listdir(folder_path)

image_path = os.path.join(folder_path, images[0])

image = Image.open(image_path)

plt.imshow(image)
plt.title(folder)
plt.axis("off")
plt.show()