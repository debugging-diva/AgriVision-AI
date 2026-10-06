import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("TensorFlow:", tf.__version__)

train_dir = "data/train"
test_dir = "data/test"

IMG_SIZE = (96, 96)
BATCH_SIZE = 64

print("Loading dataset...")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("Classes:", len(class_names))

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(AUTOTUNE)
test_data = test_data.cache().prefetch(AUTOTUNE)

print("Loading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print()
print("STARTING FAST TRAINING")
print()

model.fit(
    train_data,
    validation_data=test_data,
    epochs=3
)

os.makedirs("model", exist_ok=True)

model.save("model/plant_disease_mobilenet.keras")

print()
print("================================")
print("MODEL TRAINING COMPLETED!")
print("================================")