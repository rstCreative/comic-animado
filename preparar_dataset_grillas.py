"""
Preparar dataset de personajes a partir de las grillas/planchas subidas.
Reutiliza la misma lógica de detección de paneles del pipeline de "Cómic
Animado" (separar por márgenes de fondo + contornos), adaptada para hojas
de sprites en vez de páginas de cómic.

Salida: personajes_dataset/rstchica/*.png (+ .txt caption)
        personajes_dataset/rstchico/*.png (+ .txt caption)
        personajes_dataset/sin_clasificar/*.png  (paneles sin etiqueta legible,
                                                    para que los arrastres a mano)
"""
import os
import re
import shutil

import cv2
import numpy as np
import pytesseract
from PIL import Image

CARPETA_ENTRADA = "/mnt/user-data/uploads"
CARPETA_SALIDA = "/home/claude/personajes_dataset"
for sub in ("rstchica", "rstchico", "sin_clasificar", "descartadas"):
    os.makedirs(os.path.join(CARPETA_SALIDA, sub), exist_ok=True)

DESCRIPCION_BASE = {
    "rstchica": "long black hair, brown eyes, slim build",
    "rstchico": "short black spiky hair, brown eyes, athletic build",
}

# ──────────────────────────────────────────────────────────────────────────
# Detección de paneles (misma idea que 'detectar_paneles' de comic-animado):
# separar por el color de fondo real de la página, tomado de las esquinas.
# ──────────────────────────────────────────────────────────────────────────
def ordenar_lectura(cajas, ancho_pagina, tolerancia_fila=0.06):
    cajas = sorted(cajas, key=lambda c: c[1])
    filas = []
    for caja in cajas:
        colocada = False
        for fila in filas:
            if abs(caja[1] - fila[0][1]) < tolerancia_fila * ancho_pagina:
                fila.append(caja)
                colocada = True
                break
        if not colocada:
            filas.append([caja])
    resultado = []
    for fila in filas:
        resultado.extend(sorted(fila, key=lambda c: c[0]))
    return resultado


def detectar_paneles_con_bordes(ruta_imagen, area_min_frac=0.003):
    """Devuelve una lista de (x, y, w, h) — una por panel — o [] si la hoja
    no tiene separación real entre celdas (mosaico pegado sin gap)."""
    img = cv2.imread(ruta_imagen)
    h_total, w_total = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area_min = area_min_frac * w_total * h_total

    color_fondo = int(np.median([gris[0, 0], gris[0, -1], gris[-1, 0], gris[-1, -1]]))
    _, binaria = cv2.threshold(gris, max(color_fondo - 8, 0), 255, cv2.THRESH_BINARY_INV)
    # OJO: nada de dilatar acá — con estas grillas el gap entre celdas vecinas
    # es angosto, y dilatar (como sí conviene en páginas de cómic) fusiona
    # paneles contiguos en una sola caja gigante. Sin dilatar separa bien.
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = [cv2.boundingRect(c) for c in contornos]
    cajas = [c for c in cajas if c[2] * c[3] > area_min]

    area_total = w_total * h_total
    if len(cajas) <= 1 or (cajas and max(c[2]*c[3] for c in cajas) > 0.6 * area_total):
        return []

    return ordenar_lectura(cajas, w_total)


def dividir_grilla_fija(ruta_imagen, filas, columnas):
    """Respaldo para mosaicos edge-to-edge sin gap: división uniforme."""
    img = cv2.imread(ruta_imagen)
    h, w = img.shape[:2]
    alto_celda, ancho_celda = h // filas, w // columnas
    cajas = []
    for f in range(filas):
        for c in range(columnas):
            cajas.append((c * ancho_celda, f * alto_celda, ancho_celda, alto_celda))
    return cajas


# ──────────────────────────────────────────────────────────────────────────
# Dentro de cada panel: separar el banner de etiqueta (si existe) del cuerpo,
# y leer el texto del banner con OCR para clasificar + armar el caption.
# ──────────────────────────────────────────────────────────────────────────
def _es_fila_bandera_azul(fila_hsv):
    """Una fila de la 'bandera' de etiqueta tiene tono azul saturado
    dominante (los headers de las grillas etiquetadas)."""
    matiz, saturacion, valor = fila_hsv[:, 0], fila_hsv[:, 1], fila_hsv[:, 2]
    azul = (matiz > 95) & (matiz < 130) & (saturacion > 80) & (valor > 60)
    return azul.mean() > 0.5


def separar_banner_y_cuerpo(panel_bgr):
    """Si detecta una franja azul sólida al inicio del panel (como en las
    hojas 'rstchica/rstchico'), la devuelve separada del cuerpo + su alto en
    píxeles. Si no hay banner, devuelve el panel entero como 'cuerpo'."""
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    alto = panel_bgr.shape[0]
    fin_banner = 0
    limite_busqueda = int(alto * 0.25)  # el banner nunca ocupa más del 25% del panel
    for y in range(limite_busqueda):
        if _es_fila_bandera_azul(hsv[y]):
            fin_banner = y + 1
        elif fin_banner > 3:  # ya veníamos en banner y esta fila corta -> terminó
            break
    if fin_banner < 8:  # muy chico para ser un banner real -> no hay banner
        return None, panel_bgr
    return panel_bgr[:fin_banner], panel_bgr[fin_banner:]


def leer_texto(imagen_bgr):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    # Texto blanco sobre banner azul oscuro (como 'rstchico — 30 gestos...')
    # confunde a tesseract, que asume texto oscuro sobre fondo claro por
    # default — si el promedio de brillo es bajo, invertimos antes de leer.
    if gris.mean() < 150:
        gris = cv2.bitwise_not(gris)
    texto = pytesseract.image_to_string(gris, lang="spa+eng", config="--psm 6")
    return texto.strip()


def clasificar_por_texto(texto):
    t = texto.lower().replace(" ", "")
    if "rstchica" in t or "chica)" in t:
        return "rstchica"
    if "rstchico" in t or "chico)" in t:
        return "rstchico"
    return None


def limpiar_caption(texto):
    """De 'N. Descripción (rstchicX)' deja solo la descripción de la pose."""
    texto = re.sub(r"^\s*\d+[\.\)]?\s*", "", texto)          # saca el número inicial
    texto = re.sub(r"\(rstchic[oa]?\)?", "", texto, flags=re.I)  # saca el tag de personaje
    texto = re.sub(r"\s{2,}", " ", texto).strip(" .,-")
    return texto


# ──────────────────────────────────────────────────────────────────────────
# Procesar cada archivo subido
# ──────────────────────────────────────────────────────────────────────────
def procesar_hoja_etiquetada(nombre_archivo, ruta):
    """Hojas CON banner azul de etiqueta (turnarounds, poses, gestos):
    separa por contorno y lee texto. Dos variantes conviven en estas hojas:
    - Etiqueta el personaje en CADA celda: 'N. Pose (rstchica)' -> se lee directo.
    - Etiqueta el personaje UNA VEZ por sección, en una barra de título ancha
      ('rstchica — 30 gestos...') y las celdas de abajo solo dicen la pose,
      sin repetir el nombre -> hay que 'heredar' el personaje de la barra
      de título más reciente, en orden de lectura.
    """
    cajas = detectar_paneles_con_bordes(ruta)
    if not cajas:
        return 0, 0, []

    img = cv2.imread(ruta)
    alturas = [h for (_, _, _, h) in cajas]
    alto_mediano = float(np.median(alturas))

    conteo = {"rstchica": 0, "rstchico": 0, "sin_clasificar": 0}
    personaje_actual = None
    log_paneles = []

    for i, (x, y, w, h) in enumerate(cajas):
        panel = img[y:y+h, x:x+w]

        # Barra de título/sección: mucho más baja que una celda típica, o
        # mucho más ancha que alta -> no es una imagen de personaje, es texto.
        es_barra_titulo = (h < 0.5 * alto_mediano) or (w > 3 * h)
        if es_barra_titulo:
            texto_barra = leer_texto(panel)
            detectado = clasificar_por_texto(texto_barra)
            if detectado:
                personaje_actual = detectado
            continue  # no se guarda como imagen de entrenamiento

        banner, cuerpo = separar_banner_y_cuerpo(panel)
        texto_banner = leer_texto(banner) if banner is not None else ""
        # Prioridad: si la celda ETIQUETA el personaje explícito, usar eso.
        # Si no, heredar el de la última barra de título vista.
        personaje = clasificar_por_texto(texto_banner) or personaje_actual or "sin_clasificar"

        m = 3
        cuerpo_final = cuerpo[m:cuerpo.shape[0]-m, m:cuerpo.shape[1]-m]
        if cuerpo_final.size == 0:
            continue

        conteo[personaje] += 1
        nombre_base = f"{os.path.splitext(nombre_archivo)[0]}_p{i+1:02d}"
        ruta_salida = os.path.join(CARPETA_SALIDA, personaje, f"{nombre_base}.png")
        cv2.imwrite(ruta_salida, cuerpo_final)

        if personaje != "sin_clasificar":
            pose = limpiar_caption(texto_banner)
            token = f"{personaje}_chr"
            caption = f"{token}, {DESCRIPCION_BASE[personaje]}" + (f", {pose}" if pose else "")
            with open(ruta_salida.replace(".png", ".txt"), "w") as f:
                f.write(caption)
        log_paneles.append((personaje, texto_banner))

    return len(cajas), sum(1 for p, _ in log_paneles if p != "sin_clasificar"), log_paneles


def procesar_mosaico_sin_etiqueta(nombre_archivo, ruta, filas=4, columnas=5):
    """Hojas SIN banner (mosaico pegado, alterna personajes sin indicarlo) —
    no se puede clasificar automático de forma confiable: se recortan los
    paneles igual, pero van todos a 'sin_clasificar' para que los ordenes
    a mano (es rápido, es visualmente obvio a simple vista)."""
    img = cv2.imread(ruta)
    cajas = dividir_grilla_fija(ruta, filas, columnas)
    for i, (x, y, w, h) in enumerate(cajas):
        panel = img[y:y+h, x:x+w]
        m = 2
        panel = panel[m:panel.shape[0]-m, m:panel.shape[1]-m]
        ruta_salida = os.path.join(CARPETA_SALIDA, "sin_clasificar",
                                    f"{os.path.splitext(nombre_archivo)[0]}_p{i+1:02d}.png")
        cv2.imwrite(ruta_salida, panel)
    return len(cajas)


# ──────────────────────────────────────────────────────────────────────────
# Clasificación de archivos de entrada (según lo que ya inspeccionamos)
# ──────────────────────────────────────────────────────────────────────────
HOJAS_ETIQUETADAS = ["1000285424.png", "1000285425.png", "1000285426.png"]
HOJAS_SIN_ETIQUETA = ["1000285422.png", "1000285423.png", "1000285421.png"]
IMAGENES_LIMPIAS = {   # ya son un solo personaje, sin grilla -> se copian directo
    "1000284886.png": "rstchica",
    "1000284885.png": "rstchico",
    "1000285690.jpg": "rstchica",
}
DESCARTADAS = {
    "1000285688.jpg": "Ángulo de cámara atípico (a cuatro patas, ras de piso) — "
                       "outlier de encuadre respecto al resto del set, se deja afuera "
                       "del entrenamiento para no confundir al LoRA.",
    "acba07ca-435a-4f90-a8c9-dd753879ba57-1_all_26225.jpg": "Recorte parcial/duplicado "
                       "de contenido que ya está completo en otras imágenes del set.",
    "1000285420.png": "Hoja de diálogo con onomatopeyas — el texto está horneado "
                       "en gran parte del panel (no es solo un rótulo arriba), "
                       "así que no se puede separar limpio del arte del personaje.",
}

resumen = []

for nombre in HOJAS_ETIQUETADAS:
    ruta = os.path.join(CARPETA_ENTRADA, nombre)
    if not os.path.exists(ruta):
        continue
    total, clasificados, log = procesar_hoja_etiquetada(nombre, ruta)
    resumen.append(f"🏷️  {nombre}: {total} paneles detectados, {clasificados} clasificados por OCR "
                    f"({total - clasificados} a sin_clasificar).")

for nombre in HOJAS_SIN_ETIQUETA:
    ruta = os.path.join(CARPETA_ENTRADA, nombre)
    if not os.path.exists(ruta):
        continue
    total = procesar_mosaico_sin_etiqueta(nombre, ruta)
    resumen.append(f"🧩 {nombre}: {total} paneles recortados → sin_clasificar (sin banner, ordenar a mano).")

for nombre, personaje in IMAGENES_LIMPIAS.items():
    ruta = os.path.join(CARPETA_ENTRADA, nombre)
    if not os.path.exists(ruta):
        continue
    destino = os.path.join(CARPETA_SALIDA, personaje, nombre)
    Image.open(ruta).convert("RGB").save(destino.replace(os.path.splitext(destino)[1], ".png"))
    token = f"{personaje}_chr"
    with open(destino.replace(os.path.splitext(destino)[1], ".txt"), "w") as f:
        f.write(f"{token}, {DESCRIPCION_BASE[personaje]}")
    resumen.append(f"✅ {nombre}: imagen limpia, copiada directo a {personaje}/.")

for nombre, motivo in DESCARTADAS.items():
    ruta = os.path.join(CARPETA_ENTRADA, nombre)
    if os.path.exists(ruta):
        shutil.copy(ruta, os.path.join(CARPETA_SALIDA, "descartadas", nombre))
    resumen.append(f"🚫 {nombre}: descartada — {motivo}")

print("\n".join(resumen))
print()
for sub in ("rstchica", "rstchico", "sin_clasificar", "descartadas"):
    n = len([f for f in os.listdir(os.path.join(CARPETA_SALIDA, sub)) if f.endswith((".png", ".jpg"))])
    print(f"{sub}: {n} imágenes")
