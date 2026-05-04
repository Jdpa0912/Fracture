import argparse
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)

FRACTURE_CLASSES = ["Fractured", "Non-Fractured"]
TYPE_CLASSES = ["Comminuted", "Simple"]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m1", "--model1", type=str, default="output/fracture_detector.keras",
                help="path to fracture detector model")
ap.add_argument("-m2", "--model2", type=str, default="output/fracture_type.keras",
                help="path to fracture type model")
args = vars(ap.parse_args())

print("[INFO] loading fracture detector...")
fracture_model = load_model(args["model1"])

print("[INFO] loading fracture type model...")
type_model = load_model(args["model2"])

image = cv2.imread(args["image"])

if image is None:
    raise ValueError("Image not found. Check the path.")

orig = image.copy()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, IMG_SIZE)
rgb = rgb.astype("float32")
rgb = preprocess_input(rgb)
rgb = np.expand_dims(rgb, axis=0)

print("[INFO] predicting fracture / non-fracture...")
fracture_preds = fracture_model.predict(rgb)
fracture_idx = np.argmax(fracture_preds, axis=1)[0]
fracture_conf = fracture_preds[0][fracture_idx]
fracture_label = FRACTURE_CLASSES[fracture_idx]

if fracture_label == "Non-Fractured":
    final_label = f"Non-Fractured ({fracture_conf * 100:.2f}%)"
    color = (0, 255, 0)
else:
    print("[INFO] fracture detected, predicting type...")
    type_preds = type_model.predict(rgb)
    type_idx = np.argmax(type_preds, axis=1)[0]
    type_conf = type_preds[0][type_idx]
    type_label = TYPE_CLASSES[type_idx]

    final_label = f"Fractured - {type_label} ({type_conf * 100:.2f}%)"
    color = (0, 0, 255)

print(f"Prediction: {final_label}")

cv2.putText(
    orig,
    final_label,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    color,
    2
)

cv2.imshow("Output", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()