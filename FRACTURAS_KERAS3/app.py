import os
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "Detector de Fracturas RX"
WINDOW_GEOMETRY = "1180x760"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FRACTURE_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "fracture_detector.keras"),
    os.path.join(BASE_DIR, "output", "model.keras"),
]
MODEL_TYPE_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "fracture_type.keras"),
]
IMG_SIZE = (224, 224)

FRACTURE_CLASSES = ["Fractured", "Non-Fractured"]
TYPE_CLASSES = ["Comminuted", "Simple"]


class FractureDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry(WINDOW_GEOMETRY)
        self.minsize(1040, 680)

        self.image_path = None
        self.cv_image = None
        self.preview_image = None

        self.fracture_model = None
        self.type_model = None

        self.build_ui()
        self.load_models()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, corner_radius=20)
        self.header.grid(row=0, column=0, columnspan=2, padx=18, pady=18, sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header,
            text="Sistema de análisis de radiografías",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(18, 6), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.header,
            text="Detecta si una radiografía presenta fractura y, si existe, clasifica el tipo como simple o compuesta.",
            font=ctk.CTkFont(size=14)
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        self.left_panel = ctk.CTkFrame(self, corner_radius=20)
        self.left_panel.grid(row=1, column=0, padx=(18, 9), pady=(0, 18), sticky="nsew")
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1)

        self.left_title = ctk.CTkLabel(
            self.left_panel,
            text="Imagen cargada",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.left_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.image_container = ctk.CTkFrame(self.left_panel, corner_radius=16)
        self.image_container.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.image_container.grid_rowconfigure(0, weight=1)
        self.image_container.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.image_container,
            text="No hay imagen seleccionada",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.grid(row=0, column=0, padx=12, pady=12)

        self.button_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.button_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        self.load_button = ctk.CTkButton(
            self.button_frame,
            text="Cargar imagen",
            height=42,
            command=self.load_image
        )
        self.load_button.grid(row=0, column=0, padx=(0, 8), pady=5, sticky="ew")

        self.predict_button = ctk.CTkButton(
            self.button_frame,
            text="Analizar",
            height=42,
            command=self.predict_image
        )
        self.predict_button.grid(row=0, column=1, padx=(8, 0), pady=5, sticky="ew")

        self.right_panel = ctk.CTkFrame(self, corner_radius=20)
        self.right_panel.grid(row=1, column=1, padx=(9, 18), pady=(0, 18), sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)

        self.right_title = ctk.CTkLabel(
            self.right_panel,
            text="Resultado",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.right_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.status_card = ctk.CTkFrame(self.right_panel, corner_radius=16)
        self.status_card.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.model_status = ctk.CTkLabel(
            self.status_card,
            text="Modelos: cargando...",
            font=ctk.CTkFont(size=14),
            anchor="w",
            justify="left"
        )
        self.model_status.pack(fill="x", padx=16, pady=(16, 6))

        self.file_status = ctk.CTkLabel(
            self.status_card,
            text="Archivo: ninguno",
            font=ctk.CTkFont(size=14),
            anchor="w",
            justify="left"
        )
        self.file_status.pack(fill="x", padx=16, pady=(0, 16))

        self.result_card = ctk.CTkFrame(self.right_panel, corner_radius=16)
        self.result_card.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.result_caption = ctk.CTkLabel(
            self.result_card,
            text="Predicción principal",
            font=ctk.CTkFont(size=15)
        )
        self.result_caption.pack(anchor="w", padx=16, pady=(16, 4))

        self.result_value = ctk.CTkLabel(
            self.result_card,
            text="Esperando imagen...",
            font=ctk.CTkFont(size=30, weight="bold")
        )
        self.result_value.pack(anchor="w", padx=16, pady=(0, 10))

        self.confidence_value = ctk.CTkLabel(
            self.result_card,
            text="Confianza: --",
            font=ctk.CTkFont(size=16)
        )
        self.confidence_value.pack(anchor="w", padx=16, pady=(0, 16))

        self.type_value = ctk.CTkLabel(
            self.result_card,
            text="Tipo de fractura: --",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.type_value.pack(anchor="w", padx=16, pady=(0, 16))

        self.details_card = ctk.CTkFrame(self.right_panel, corner_radius=16)
        self.details_card.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        self.details_title = ctk.CTkLabel(
            self.details_card,
            text="Detalles",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.details_title.pack(anchor="w", padx=16, pady=(16, 8))

        self.details_text = ctk.CTkTextbox(self.details_card, height=240, corner_radius=12)
        self.details_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.details_text.insert(
            "0.0",
            "Esta versión usa dos etapas:\n\n"
            "1. Detector de fractura: Fractured / Non-Fractured\n"
            "2. Clasificador de tipo: Simple / Comminuted\n\n"
            "Si no hay fractura, no se analiza el tipo."
        )
        self.details_text.configure(state="disabled")

    def _load_first_existing_model(self, candidate_paths):
        existing_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if existing_path is None:
            return None, None
        return load_model(existing_path), existing_path

    def load_models(self):
        messages = []

        try:
            self.fracture_model, fracture_model_path = self._load_first_existing_model(MODEL_FRACTURE_CANDIDATES)
            if self.fracture_model is not None:
                messages.append(f"Detector de fractura: OK\nRuta: {fracture_model_path}")
            else:
                messages.append(
                    "Detector de fractura: NO encontrado\n"
                    f"Rutas revisadas: {', '.join(MODEL_FRACTURE_CANDIDATES)}"
                )
        except Exception as e:
            messages.append(f"Detector de fractura: ERROR\n{str(e)}")

        try:
            self.type_model, type_model_path = self._load_first_existing_model(MODEL_TYPE_CANDIDATES)
            if self.type_model is not None:
                messages.append(f"Clasificador de tipo: OK\nRuta: {type_model_path}")
            else:
                messages.append(
                    "Clasificador de tipo: NO encontrado\n"
                    f"Rutas revisadas: {', '.join(MODEL_TYPE_CANDIDATES)}"
                )
        except Exception as e:
            messages.append(f"Clasificador de tipo: ERROR\n{str(e)}")

        self.model_status.configure(text="\n\n".join(messages))

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar radiografía",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "No se pudo abrir la imagen seleccionada.")
            return

        self.image_path = path
        self.cv_image = image
        self.file_status.configure(text=f"Archivo: {os.path.basename(path)}")

        self.show_preview(path)

        self.result_value.configure(text="Imagen cargada", text_color=("white", "white"))
        self.confidence_value.configure(text="Confianza: --")
        self.type_value.configure(text="Tipo de fractura: --")

        self.details_text.configure(state="normal")
        self.details_text.delete("0.0", "end")
        self.details_text.insert("0.0", "Imagen cargada correctamente.\nPresiona 'Analizar' para obtener el resultado.")
        self.details_text.configure(state="disabled")

    def show_preview(self, path):
        pil_image = Image.open(path)
        pil_image.thumbnail((480, 480))
        self.preview_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=pil_image.size
        )
        self.image_label.configure(text="", image=self.preview_image)

    def preprocess_image(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, IMG_SIZE)
        image_rgb = image_rgb.astype("float32")
        image_rgb = preprocess_input(image_rgb)
        image_rgb = np.expand_dims(image_rgb, axis=0)
        return image_rgb

    def predict_image(self):
        if self.cv_image is None:
            messagebox.showwarning("Aviso", "Primero debes cargar una imagen.")
            return

        if self.fracture_model is None:
            messagebox.showerror("Error", "El modelo detector no está cargado.")
            return

        try:
            input_image = self.preprocess_image(self.cv_image)

            fracture_preds = self.fracture_model.predict(input_image, verbose=0)
            fracture_idx = np.argmax(fracture_preds, axis=1)[0]
            fracture_conf = float(fracture_preds[0][fracture_idx])
            fracture_label = FRACTURE_CLASSES[fracture_idx]

            if fracture_label == "Non-Fractured":
                self.result_value.configure(text="No fracturada", text_color="#57C785")
                self.confidence_value.configure(text=f"Confianza detector: {fracture_conf * 100:.2f}%")
                self.type_value.configure(text="Tipo de fractura: No aplica")

                self.details_text.configure(state="normal")
                self.details_text.delete("0.0", "end")
                self.details_text.insert(
                    "0.0",
                    f"Resultado del detector: {fracture_label}\n"
                    f"Confianza: {fracture_conf * 100:.2f}%\n\n"
                    "La imagen fue clasificada como no fracturada.\n"
                    "Por eso no se ejecutó la clasificación del tipo de fractura."
                )
                self.details_text.configure(state="disabled")
                return

            self.result_value.configure(text="Fracturada", text_color="#FF5A5F")
            self.confidence_value.configure(text=f"Confianza detector: {fracture_conf * 100:.2f}%")

            if self.type_model is None:
                self.type_value.configure(text="Tipo de fractura: modelo no disponible")
                self.details_text.configure(state="normal")
                self.details_text.delete("0.0", "end")
                self.details_text.insert(
                    "0.0",
                    f"Resultado del detector: {fracture_label}\n"
                    f"Confianza detector: {fracture_conf * 100:.2f}%\n\n"
                    "Se detectó fractura, pero no se encontró el modelo de tipo.\n"
                    f"Debes entrenar y guardar en una de estas rutas: {', '.join(MODEL_TYPE_CANDIDATES)}"
                )
                self.details_text.configure(state="disabled")
                return

            type_preds = self.type_model.predict(input_image, verbose=0)
            type_idx = np.argmax(type_preds, axis=1)[0]
            type_conf = float(type_preds[0][type_idx])
            type_label = TYPE_CLASSES[type_idx]

            type_label_es = "Compuesta" if type_label == "Comminuted" else "Simple"

            self.type_value.configure(
                text=f"Tipo de fractura: {type_label_es} ({type_conf * 100:.2f}%)"
            )

            self.details_text.configure(state="normal")
            self.details_text.delete("0.0", "end")
            self.details_text.insert(
                "0.0",
                f"Resultado del detector: {fracture_label}\n"
                f"Confianza detector: {fracture_conf * 100:.2f}%\n\n"
                f"Tipo de fractura predicho: {type_label_es}\n"
                f"Confianza tipo: {type_conf * 100:.2f}%\n\n"
                "La imagen fue detectada como fracturada y luego clasificada "
                "por el segundo modelo de tipo de fractura."
            )
            self.details_text.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Error durante la predicción", str(e))


if __name__ == "__main__":
    app = FractureDetectorApp()
    app.mainloop()
