from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array, load_img
from imutils import paths
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("[INFO] loading fracture detector...")
model = load_model("output/fracture_detector.keras")

IMG_SIZE = (224, 224)
CLASS_TO_INDEX = {
    "Fractured": 0,
    "Non-Fractured": 1
}
CLASS_NAMES = ["Fractured", "Non-Fractured"]

def load_images_from_folder(folder, class_name):
    imagePaths = sorted(list(paths.list_images(folder)))
    data = []
    labels = []

    for imagePath in imagePaths:
        image = load_img(imagePath, target_size=IMG_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(CLASS_TO_INDEX[class_name])

    return np.array(data, dtype="float32"), np.array(labels)

print("[INFO] loading test images...")

data_fractured, labels_fractured = load_images_from_folder("Test/Fractured", "Fractured")
data_nonfractured, labels_nonfractured = load_images_from_folder("Test/Non-Fractured", "Non-Fractured")

X_test = np.vstack([data_fractured, data_nonfractured])
y_test = np.hstack([labels_fractured, labels_nonfractured])

print("[INFO] predicting...")
preds = model.predict(X_test, batch_size=32)
y_pred = np.argmax(preds, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("[INFO] Confusion Matrix:")
print(cm)
print(f"[INFO] Accuracy: {acc:.4f}")