import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

MODELO_PATH = Path(__file__).parent / "modelo_productos.pt"
INTERVALO_MS = 30
CADA_N_FRAMES = 8


class Escaner:
    def __init__(self, raiz, modelo):
        self.raiz = raiz
        self.modelo = modelo
        self.camara = None
        self.foto = None
        self.contador = 0
        self.texto = tk.StringVar(value="Inicia la cámara o abre una imagen.")

        raiz.title("Escáner de productos")
        raiz.configure(bg="#f0fdf4")
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

        tk.Label(
            raiz, text="🛒 Reconocimiento de productos", bg="#15803d", fg="white",
            font=("Segoe UI", 16, "bold"), pady=10,
        ).pack(fill="x")

        self.pantalla = tk.Label(
            raiz, text="Sin imagen", bg="#14532d", fg="white",
            font=("Segoe UI", 12), padx=200, pady=140,
        )
        self.pantalla.pack(padx=12, pady=12)

        botones = tk.Frame(raiz, bg="#f0fdf4")
        botones.pack(pady=4)
        for texto, comando in [
            ("Iniciar cámara", self.iniciar_camara),
            ("Detener cámara", self.detener_camara),
            ("Abrir imagen...", self.abrir_imagen),
        ]:
            tk.Button(
                botones, text=texto, command=comando, bg="#16a34a", fg="white",
                activebackground="#15803d", activeforeground="white",
                font=("Segoe UI", 11, "bold"), relief="flat", padx=14, pady=6,
            ).pack(side="left", padx=6)

        tk.Label(
            raiz, textvariable=self.texto, bg="#dcfce7", fg="#14532d",
            font=("Segoe UI", 14, "bold"), justify="left", padx=16, pady=12,
        ).pack(fill="x", padx=12, pady=12)

    def iniciar_camara(self):
        if self.camara is not None:
            return
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        camara = cv2.VideoCapture(0, backend)
        if not camara.isOpened():
            camara.release()
            messagebox.showerror("Cámara", "No se pudo abrir la cámara.")
            return
        self.camara = camara
        self.bucle()

    def detener_camara(self):
        if self.camara is not None:
            self.camara.release()
            self.camara = None

    def bucle(self):
        if self.camara is None:
            return
        ok, frame = self.camara.read()
        if ok:
            self.contador += 1
            if self.contador % CADA_N_FRAMES == 1:
                self.reconocer(frame)
            self.mostrar(frame)
        self.raiz.after(INTERVALO_MS, self.bucle)

    def reconocer(self, imagen_bgr):
        r = self.modelo.predict(imagen_bgr, verbose=False)[0]
        indices = r.probs.top5[:3]
        confianzas = r.probs.top5conf[:3].tolist()
        lineas = [
            f"{r.names[i].replace('_', ' ')}: {c * 100:.1f}%"
            for i, c in zip(indices, confianzas)
        ]
        self.texto.set("\n".join(lineas))

    def mostrar(self, imagen_bgr):
        rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((640, 480))
        self.foto = ImageTk.PhotoImage(img)
        self.pantalla.configure(image=self.foto, text="", padx=0, pady=0)

    def abrir_imagen(self):
        self.detener_camara()
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp")]
        )
        if not ruta:
            return
        datos = np.fromfile(ruta, dtype=np.uint8)
        imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        if imagen is None:
            messagebox.showerror("Imagen", "No se pudo leer la imagen.")
            return
        self.reconocer(imagen)
        self.mostrar(imagen)

    def cerrar(self):
        self.detener_camara()
        self.raiz.destroy()


def main():
    raiz = tk.Tk()
    if not MODELO_PATH.exists():
        messagebox.showerror(
            "Modelo",
            "No se encontró modelo_productos.pt.\nEntrena primero desde la app web.",
        )
        raiz.destroy()
        return
    modelo = YOLO(str(MODELO_PATH))
    Escaner(raiz, modelo)
    raiz.mainloop()


if __name__ == "__main__":
    main()