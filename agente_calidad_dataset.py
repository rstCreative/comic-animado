"""
Agente de control de calidad del dataset
==========================================
Recorre las carpetas de personajes (personajes_dataset/rstchica,
personajes_dataset/rstchico, o cualquier carpeta que le indiques) y revisa
cada imagen con los mismos 3 chequeos del generador:

  1. Manos de más       (mediapipe HandLandmarker)
  2. Cabezas de más     (cascada LBP específica para caras anime)
  3. Piernas/pies de más (silueta recortada con rembg)

Las imágenes que pasan los 3 chequeos se dejan donde están. Las que no,
se MUEVEN (junto con su .txt de caption si tiene) a una carpeta
'revisar_manual/<personaje>/' aparte — así el dataset "limpio" que le vas a
dar al entrenamiento no incluye nada sospechoso, sin que tengas que revisar
las 100+ imágenes una por una: solo revisás las que quedaron en
revisar_manual/.

IMPORTANTE — qué hace y qué NO hace:
- Organiza (mueve lo sospechoso afuera). NO corrige/repinta la imagen —
  no existe una forma confiable de "arreglar" un brazo de más
  automáticamente sin riesgo de arruinar el dibujo. Si algo cae en
  revisar_manual/, las opciones reales son: descartarla, recortarla a mano
  si el defecto está en un borde, o no usarla para entrenar.
- Es heurístico y conservador a propósito (mismo criterio que en el
  generador): solo marca algo como sospechoso si detecta CLARAMENTE MÁS
  cabezas/manos/piernas de las esperadas. Puede dejar pasar algún defecto
  sutil (dedos raros, proporciones extrañas) — esto no reemplaza mirar el
  dataset vos, pero sí filtra el caso más grave (duplicación de partes del
  cuerpo) sin que tengas que hacerlo a mano imagen por imagen.

Uso:
    python agente_calidad_dataset.py [carpeta_dataset]
    (por default usa ./personajes_dataset)
"""
import os
import sys
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

CARPETA_DATASET = sys.argv[1] if len(sys.argv) > 1 else "personajes_dataset"
CARPETA_TRABAJO = os.path.join(CARPETA_DATASET, "_modelos_qc")
os.makedirs(CARPETA_TRABAJO, exist_ok=True)

UMBRAL_MANOS = 2
UMBRAL_CABEZAS = 1
UMBRAL_PIERNAS = 2

# Subcarpetas que el agente ignora (ya son de revisión/descartes, no de personaje)
CARPETAS_IGNORAR = {"sin_clasificar", "descartadas", "revisar_manual", "_modelos_qc"}


# ──────────────────────────────────────────────────────────────────────────
# Manos — mediapipe HandLandmarker
# ──────────────────────────────────────────────────────────────────────────
_detector_manos = None
_RUTA_MODELO_MANOS = os.path.join(CARPETA_TRABAJO, "hand_landmarker.task")
_URL_MODELO_MANOS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _cargar_detector_manos():
    global _detector_manos
    if _detector_manos is not None:
        return _detector_manos
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

        if not os.path.exists(_RUTA_MODELO_MANOS):
            subprocess.run(["wget", "-q", "-O", _RUTA_MODELO_MANOS, _URL_MODELO_MANOS], check=True)
        opciones = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_RUTA_MODELO_MANOS),
            num_hands=6, min_hand_detection_confidence=0.15,
        )
        _detector_manos = HandLandmarker.create_from_options(opciones)
    except Exception as e:
        print(f"⚠️ Detector de manos no disponible: {e}")
        _detector_manos = False
    return _detector_manos


def contar_manos(imagen_pil):
    import mediapipe as mp
    detector = _cargar_detector_manos()
    if detector is False:
        return None
    arr = np.array(imagen_pil.convert("RGB"))
    mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    resultado = detector.detect(mp_imagen)
    return len(resultado.hand_landmarks)


# ──────────────────────────────────────────────────────────────────────────
# Cabezas — cascada LBP anime (nagadomi/lbpcascade_animeface)
# ──────────────────────────────────────────────────────────────────────────
_cascada_caras = None
_RUTA_CASCADA = os.path.join(CARPETA_TRABAJO, "lbpcascade_animeface.xml")
_URL_CASCADA = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"


def _cargar_cascada_caras():
    global _cascada_caras
    if _cascada_caras is not None:
        return _cascada_caras
    try:
        if not os.path.exists(_RUTA_CASCADA):
            subprocess.run(["wget", "-q", "-O", _RUTA_CASCADA, _URL_CASCADA], check=True)
        cascada = cv2.CascadeClassifier(_RUTA_CASCADA)
        if cascada.empty():
            raise RuntimeError("archivo de cascada corrupto")
        _cascada_caras = cascada
    except Exception as e:
        print(f"⚠️ Detector de caras anime no disponible: {e}")
        _cascada_caras = False
    return _cascada_caras


def contar_cabezas(imagen_pil):
    cascada = _cargar_cascada_caras()
    if cascada is False:
        return None
    arr = np.array(imagen_pil.convert("RGB"))
    gris = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gris = cv2.equalizeHist(gris)
    lado_min = max(20, int(min(arr.shape[0], arr.shape[1]) * 0.05))
    caras = cascada.detectMultiScale(gris, scaleFactor=1.05, minNeighbors=4, minSize=(lado_min, lado_min))
    return len(caras)


# ──────────────────────────────────────────────────────────────────────────
# Piernas — silueta recortada con rembg
# ──────────────────────────────────────────────────────────────────────────
_sesion_rembg = None


def _cargar_rembg():
    global _sesion_rembg
    if _sesion_rembg is None:
        from rembg import new_session
        _sesion_rembg = new_session("isnet-anime")
    return _sesion_rembg


def contar_bloques_piernas(imagen_pil):
    try:
        from rembg import remove
        sesion = _cargar_rembg()
        recorte = remove(imagen_pil.convert("RGB"), session=sesion)
    except Exception as e:
        print(f"⚠️ Detector de piernas (rembg) no disponible: {e}")
        return None
    arr = np.array(recorte)
    alpha = arr[:, :, 3]
    h, _ = alpha.shape
    y0, y1 = int(h * 0.90), int(h * 0.96)
    franja = (alpha[y0:y1] > 128).astype(np.uint8)
    if franja.size == 0:
        return None
    columna = franja.max(axis=0)
    # Cierre morfológico horizontal: ignora huecos de 1-4px (ruido de rembg),
    # que si no se cierra parte una sola pierna en dos "bloques" (falso
    # positivo real que encontramos probando esto contra el dataset).
    columna_2d = (columna * 255).reshape(1, -1).astype(np.uint8)
    columna_cerrada = cv2.morphologyEx(columna_2d, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    columna_final = (columna_cerrada[0] > 0).astype(int)
    transiciones = np.diff(columna_final)
    return int((transiciones == 1).sum()) + (1 if columna_final[0] else 0)


# ──────────────────────────────────────────────────────────────────────────
# Chequeo combinado + organización de carpetas
# ──────────────────────────────────────────────────────────────────────────
def revisar_imagen(ruta_imagen):
    img = Image.open(ruta_imagen)
    motivos = []

    manos = contar_manos(img)
    if manos is not None and manos > UMBRAL_MANOS:
        motivos.append(f"{manos} manos")

    cabezas = contar_cabezas(img)
    if cabezas is not None and cabezas > UMBRAL_CABEZAS:
        motivos.append(f"{cabezas} caras")

    piernas = contar_bloques_piernas(img)
    if piernas is not None and piernas > UMBRAL_PIERNAS:
        motivos.append(f"{piernas} bloques de pierna/pie")

    return motivos


def procesar_personaje(nombre_personaje):
    carpeta_personaje = os.path.join(CARPETA_DATASET, nombre_personaje)
    carpeta_revisar = os.path.join(CARPETA_DATASET, "revisar_manual", nombre_personaje)
    os.makedirs(carpeta_revisar, exist_ok=True)

    archivos = [f for f in sorted(os.listdir(carpeta_personaje)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    total, movidas = len(archivos), 0

    for i, nombre_archivo in enumerate(archivos):
        ruta = os.path.join(carpeta_personaje, nombre_archivo)
        print(f"[{nombre_personaje}] Revisando {i+1}/{total}: {nombre_archivo}...", flush=True)
        motivos = revisar_imagen(ruta)

        if motivos:
            movidas += 1
            print(f"   ⚠️ Sospechosa ({', '.join(motivos)}) → movida a revisar_manual/{nombre_personaje}/")
            shutil.move(ruta, os.path.join(carpeta_revisar, nombre_archivo))
            ruta_caption = ruta.replace(os.path.splitext(ruta)[1], ".txt")
            if os.path.exists(ruta_caption):
                shutil.move(ruta_caption, os.path.join(carpeta_revisar, os.path.basename(ruta_caption)))

    return total, movidas


if __name__ == "__main__":
    if not os.path.isdir(CARPETA_DATASET):
        print(f"❌ No existe la carpeta '{CARPETA_DATASET}'.")
        sys.exit(1)

    personajes = [d for d in sorted(os.listdir(CARPETA_DATASET))
                  if os.path.isdir(os.path.join(CARPETA_DATASET, d)) and d not in CARPETAS_IGNORAR]

    if not personajes:
        print(f"❌ No encontré subcarpetas de personaje dentro de '{CARPETA_DATASET}'.")
        sys.exit(1)

    print(f"Personajes encontrados: {', '.join(personajes)}\n")
    resumen = []
    for nombre in personajes:
        total, movidas = procesar_personaje(nombre)
        resumen.append((nombre, total, movidas))

    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    for nombre, total, movidas in resumen:
        print(f"{nombre}: {total} revisadas, {movidas} movidas a revisar_manual/ ({total - movidas} quedaron limpias)")
    print("\nRevisá manualmente lo que quedó en 'revisar_manual/' antes de decidir si lo descartás,"
          " lo recortás, o lo dejás pasar igual.")
