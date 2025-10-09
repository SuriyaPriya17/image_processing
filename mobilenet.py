# -------------------------------------
# 1. Mount Google Drive
# -------------------------------------
from google.colab import drive
import os

# Mount Drive
drive.mount('/content/drive')

# Create a folder in Drive to store your model
save_dir = '/content/drive/MyDrive/models'
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------
# 2. Setup TensorFlow
# -------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

# Normalize images to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0

resize_layer = tf.keras.Sequential([
    layers.Resizing(224, 224)
])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .batch(32).map(lambda x, y: (resize_layer(x), y))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
    .batch(32).map(lambda x, y: (resize_layer(x), y))

# -------------------------------------
# 4. Transfer Learning with MObileNet
# -------------------------------------
 
base_resnet = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_resnet.trainable = False

# Build full model
mobilenet_model = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# Compile model
mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------
# 5. Train the model
# -------------------------------------
print("Training MobileNetV2 (feature extraction)...")
resnet_model.fit(train_ds, validation_data=val_ds, epochs=3)

# -------------------------------------
# 6. Fine-tuning (unfreeze last few layers)
# -------------------------------------
base_resnet.trainable = True
for layer in base_resnet.layers[:-30]:
    layer.trainable = False

mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning MobileNetV2...")
mobilenet_model.fit(train_ds, validation_data=val_ds, epochs=2)
save_path = os.path.join(save_dir, "mobilenetV2.keras")
mobilenet_model.save(save_path)