import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "model/plant_disease_mobilenet.keras"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(96, 96),
    batch_size=64,
    shuffle=False
)

class_names = test_data.class_names

print()
print("Evaluating MobileNetV2...")
print()

loss, accuracy = model.evaluate(test_data)

print()
print("==============================")
print("MODEL EVALUATION")
print("==============================")
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("==============================")
print()
print("Classes:")
for i, name in enumerate(class_names):
    print(i, "-", name)