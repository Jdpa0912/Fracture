from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array, load_img
from imutils import paths
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("[INFO] loading fracture type model...")
model = load_model("output/fracture_type.keras")

IMG_SIZE = (224, 224)
CLASS_TO_INDEX = {
    "Comminuted": 0,
    "Simple": 1
}
CLASS_NAMES = ["Comminuted", "Simple"]

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

print("[INFO] loading type test images...")

data_simple, labels_simple = load_images_from_folder("TypeTest/Simple", "Simple")
data_comminuted, labels_comminuted = load_images_from_folder("TypeTest/Comminuted", "Comminuted")

X_test = np.vstack([data_simple, data_comminuted])
y_test = np.hstack([labels_simple, labels_comminuted])

print("[INFO] predicting type...")
preds = model.predict(X_test, batch_size=32)
y_pred = np.argmax(preds, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("[INFO] Confusion Matrix:")
print(cm)
print(f"[INFO] Accuracy: {acc:.4f}")