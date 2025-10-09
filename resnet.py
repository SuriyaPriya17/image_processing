
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

# -------------------------------------
# 3. Load the CIFAR-10 Dataset
# -------------------------------------
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

# Normalize images to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0

# Resize images to 224x224 for ResNet
resize_layer = tf.keras.Sequential([
    layers.Resizing(224, 224)
])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .batch(32).map(lambda x, y: (resize_layer(x), y))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
    .batch(32).map(lambda x, y: (resize_layer(x), y))

# -------------------------------------
# 4. Transfer Learning with ResNet50
# -------------------------------------
base_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_resnet.trainable = False
resnet_model = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")  
])

# Compile model
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------
# 5. Train the model
# -------------------------------------
print("Training ResNet50 (feature extraction)...")
resnet_model.fit(train_ds, validation_data=val_ds, epochs=3)

# -------------------------------------
# 6. Fine-tuning (unfreeze last few layers)
# -------------------------------------
base_resnet.trainable = True
for layer in base_resnet.layers[:-30]:
    layer.trainable = False

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning ResNet50...")
resnet_model.fit(train_ds, validation_data=val_ds, epochs=2)

# -------------------------------------
# 7. Save model to Google Drive
# -------------------------------------
save_path_h5 = os.path.join(save_dir, "resnet50_cifar10_model.h5")
save_path_keras = os.path.join(save_dir, "resnet50_cifar10_model.keras")

# Save both formats
resnet_model.save(save_path_h5)
resnet_model.save(save_path_keras)

print(f"Model successfully saved to:\n{save_path_h5}\n{save_path_keras}")
from tensorflow.keras.preprocessing import image

img_path = "aero.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # batch dimension

pred_probs = resnet_model.predict(img_array)
pred_class = np.argmax(pred_probs, axis=1)[0]

print("Predicted class:", pred_class)

