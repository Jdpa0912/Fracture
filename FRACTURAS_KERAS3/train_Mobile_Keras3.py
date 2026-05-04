import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img, img_to_array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Inicio")

IMG_SIZE = (224, 224)
DATASET_DIR = "Train"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "fracture_detector.keras")
PLOT_PATH = os.path.join(OUTPUT_DIR, "fracture_detector_plot.png")

print("[INFO] Loading images...")
data = []
labels = []

image_paths = sorted(list(paths.list_images(DATASET_DIR)))

for imagePath in image_paths:
    image = load_img(imagePath, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)

    label = imagePath.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

trainY = keras.utils.to_categorical(trainY, 2)
testY = keras.utils.to_categorical(testY, 2)

print("[INFO] Classes:", lb.classes_)

print("[INFO] Preparing model...")

baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

baseModel.trainable = False

x = baseModel.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = models.Model(inputs=baseModel.input, outputs=outputs)

optimizer = Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

EPOCHS = 20
BS = 16

print("[INFO] Training...")

history = model.fit(
    trainX,
    trainY,
    validation_data=(testX, testY),
    epochs=EPOCHS,
    batch_size=BS,
    callbacks=callbacks
)

print("[INFO] Evaluating...")
predictions = model.predict(testX, batch_size=BS)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(testY, axis=1)

print(classification_report(true_classes, predicted_classes, target_names=lb.classes_))

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(MODEL_PATH)

plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Fracture Detector - Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"[INFO] Model saved to: {MODEL_PATH}")
print(f"[INFO] Plot saved to: {PLOT_PATH}")