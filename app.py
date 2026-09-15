"""
Cómic Animado — Interfaz simple (Gradio)
=========================================
Sube una página de cómic, presiona "Generar Video" y obtén un .mp4 en 4K
con diálogos narrados por voces naturales (XTTS-v2) y animación suave.

100% open source. Corre igual dentro de Google Colab que en tu propia
PC/servidor con GPU, o en una VM en la nube (RunPod, Vast.ai, etc.).

Uso:
    pip install -r requirements.txt
    python app.py
Se abrirá un link local (y uno público si compartes) en tu navegador.
"""

import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
import torch.hub
# Evita el prompt interactivo "¿confías en este repositorio? (y/N)" que bloquea
# todo en un entorno sin consola (como aquí) — aplica a TODAS las descargas de
# torch.hub, incluidas las que MiDaS dispara internamente para sus submódulos.
torch.hub._check_repo_is_trusted = lambda *a, **k: None
import gradio as gr
from PIL import Image, ImageEnhance

# ──────────────────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────────────────
CARPETA_TRABAJO = "trabajo"
ANCHO_4K, ALTO_4K = 3840, 2160
ANCHO_BASE, ALTO_BASE = 1920, 1080  # lienzo uniforme para el video base (16:9)

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # evita bloqueo por licencia de XTTS-v2

_tts = None            # se carga una sola vez, la primera vez que se usa
_ocr_reader = None


# ──────────────────────────────────────────────────────────────────────────
# Paso 0: Real-ESRGAN en PyTorch puro (sin basicsr, sin ejecutables externos)
#
# 'basicsr'/'realesrgan' (pip) no instalan en Python 3.13 (Colab actual): dependen
# de 'distutils', que Python eliminó del todo en esta versión — no es arreglable
# con configuración, el paquete en sí ya no es compatible. El ejecutable Vulkan
# tampoco sirve porque Colab no expone Vulkan para la GPU, solo CUDA.
# Solución: la misma arquitectura de red (RRDBNet), implementada aquí directo
# con torch.nn (igual que usan las voces), corriendo sobre CUDA sin intermediarios.
# ──────────────────────────────────────────────────────────────────────────
import torch.nn as nn
import torch.nn.functional as Fnn

_modelo_esrgan = None
_URL_PESOS_ESRGAN = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
_RUTA_PESOS_ESRGAN = os.path.join(CARPETA_TRABAJO, "RealESRGAN_x4plus_anime_6B.pth")


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Misma arquitectura que usa Real-ESRGAN, sin depender del paquete 'basicsr'."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(Fnn.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(Fnn.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _cargar_modelo_esrgan():
    global _modelo_esrgan
    if _modelo_esrgan is not None:
        return _modelo_esrgan

    os.makedirs(CARPETA_TRABAJO, exist_ok=True)
    if not os.path.exists(_RUTA_PESOS_ESRGAN):
        subprocess.run(["wget", "-nc", _URL_PESOS_ESRGAN, "-O", _RUTA_PESOS_ESRGAN], check=True)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = RRDBNet(num_block=6)
    estado = torch.load(_RUTA_PESOS_ESRGAN, map_location=dispositivo)
    modelo.load_state_dict(estado["params_ema"] if "params_ema" in estado else estado)
    modelo.eval().to(dispositivo)
    _modelo_esrgan = modelo
    return _modelo_esrgan


def escalar_con_esrgan(imagen_pil, tamano_tile=512, relleno=16):
    """Escala una imagen PIL x4 usando la GPU (CUDA) por bloques, para no saturar memoria."""
    modelo = _cargar_modelo_esrgan()
    dispositivo = next(modelo.parameters()).device

    arr = np.array(imagen_pil.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dispositivo)
    _, _, h, w = tensor.shape
    salida = torch.zeros((1, 3, h * 4, w * 4), device=dispositivo)

    with torch.no_grad():
        for y in range(0, h, tamano_tile):
            for x in range(0, w, tamano_tile):
                y0, x0 = max(0, y - relleno), max(0, x - relleno)
                y1, x1 = min(h, y + tamano_tile + relleno), min(w, x + tamano_tile + relleno)
                bloque = tensor[:, :, y0:y1, x0:x1]
                resultado = modelo(bloque)

                oy0, ox0 = (y - y0) * 4, (x - x0) * 4
                oy1 = oy0 + min(tamano_tile, h - y) * 4
                ox1 = ox0 + min(tamano_tile, w - x) * 4
                salida[:, :, y*4:min(h,y+tamano_tile)*4, x*4:min(w,x+tamano_tile)*4] = \
                    resultado[:, :, oy0:oy1, ox0:ox1]

    salida = salida.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((salida * 255).round().astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────
# Paso 1: preprocesamiento — más resolución + color más vivo (sin alterar nada)
# ──────────────────────────────────────────────────────────────────────────
def preprocesar_imagen(ruta_entrada, factor_saturacion=1.15, limite_lado_px=2400):
    img_pil = Image.open(ruta_entrada).convert("RGB")
    base, _ = os.path.splitext(ruta_entrada)

    if max(img_pil.size) < limite_lado_px:
        img_pil = escalar_con_esrgan(img_pil)  # x4 real, vía GPU

    img_pil = ImageEnhance.Color(img_pil).enhance(factor_saturacion)
    ruta_salida = f"{base}_preprocesada.png"
    img_pil.save(ruta_salida)
    return ruta_salida


# ──────────────────────────────────────────────────────────────────────────
# Paso 2: detección y recorte automático de paneles
# ──────────────────────────────────────────────────────────────────────────
def ordenar_lectura(cajas, ancho_pagina, tolerancia_fila=0.15):
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


def _filtrar_cajas_anidadas(cajas):
    """Quita cajas que están completamente dentro de otra más grande
    (común cuando Canny detecta el borde interior Y exterior del mismo trazo)."""
    cajas = sorted(cajas, key=lambda c: c[2] * c[3], reverse=True)
    resultado = []
    for x, y, w, h in cajas:
        contenida = False
        for xr, yr, wr, hr in resultado:
            if x >= xr - 5 and y >= yr - 5 and x + w <= xr + wr + 5 and y + h <= yr + hr + 5:
                contenida = True
                break
        if not contenida:
            resultado.append((x, y, w, h))
    return resultado


def _detectar_paneles_por_bordes(gris, area_min):
    """Respaldo para cuando los paneles están pegados sin margen blanco entre
    ellos: en vez de buscar el fondo, busca directamente el trazo/borde propio
    de cada viñeta."""
    bordes = cv2.Canny(gris, 50, 150)
    bordes = cv2.dilate(bordes, np.ones((5, 5), np.uint8), iterations=2)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cajas = []
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue
        aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(aprox) >= 4:  # forma aproximadamente rectangular
            cajas.append((x, y, w, h))
    return _filtrar_cajas_anidadas(cajas)


def detectar_paneles(ruta_imagen, area_min_frac=0.02):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
    h_total, w_total = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area_min = area_min_frac * w_total * h_total

    # Método 1: separar por márgenes claros entre paneles (color de fondo real
    # de la página, tomado de las esquinas — no siempre es blanco puro).
    color_fondo = int(np.median([gris[0, 0], gris[0, -1], gris[-1, 0], gris[-1, -1]]))
    _, binaria = cv2.threshold(gris, max(color_fondo - 15, 0), 255, cv2.THRESH_BINARY_INV)
    binaria = cv2.dilate(binaria, np.ones((15, 15), np.uint8), iterations=2)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = [cv2.boundingRect(c) for c in contornos if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > area_min]

    # Si esto solo encontró 1 caja gigante (== casi toda la página), los paneles
    # probablemente están pegados sin margen — se intenta el método de bordes.
    area_total = w_total * h_total
    if len(cajas) <= 1 and (not cajas or cajas[0][2] * cajas[0][3] > 0.7 * area_total):
        cajas_bordes = _detectar_paneles_por_bordes(gris, area_min)
        if len(cajas_bordes) > len(cajas):
            cajas = cajas_bordes

    cajas = ordenar_lectura(cajas, w_total)

    paneles = []
    for i, (x, y, w, h) in enumerate(cajas):
        salida = os.path.join(CARPETA_TRABAJO, f"panel_{i+1:02d}.png")
        cv2.imwrite(salida, img[y:y+h, x:x+w])
        paneles.append(salida)

    if not paneles:  # último recurso: la página completa como un solo panel
        salida = os.path.join(CARPETA_TRABAJO, "panel_01.png")
        cv2.imwrite(salida, img)
        paneles = [salida]
        print("⚠️ No se pudieron separar los paneles automáticamente; se usó la página completa. "
              "El layout de esta página puede ser demasiado irregular para la detección automática.")

    return paneles


# ──────────────────────────────────────────────────────────────────────────
# Paso 3: diálogos + personajes automáticos (OCR + reconocimiento facial)
# ──────────────────────────────────────────────────────────────────────────
def _cargar_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
    return _ocr_reader


def generar_guion(paneles):
    import face_recognition

    reader = _cargar_ocr()
    identidades_conocidas = []
    TOLERANCIA_ROSTRO = 0.55

    def identificar_personaje(encoding):
        for enc_conocido, id_personaje in identidades_conocidas:
            if face_recognition.face_distance([enc_conocido], encoding)[0] < TOLERANCIA_ROSTRO:
                return id_personaje
        nuevo_id = f"personaje_{len(identidades_conocidas) + 1}"
        identidades_conocidas.append((encoding, nuevo_id))
        return nuevo_id

    guion = []
    ultimo_personaje = None
    for i, p in enumerate(paneles):
        print(f"[3/6] Leyendo diálogo/rostro del panel {i+1}/{len(paneles)}...", flush=True)
        texto = " ".join(reader.readtext(p, detail=0, paragraph=True)).strip()

        img = face_recognition.load_image_file(p)
        ubicaciones = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, ubicaciones)

        alto_panel, ancho_panel = img.shape[0], img.shape[1]
        personaje = ultimo_personaje or "personaje_1"
        cara_frac = 0.0
        if encodings:
            areas = [(b[2]-b[0]) * (b[1]-b[3]) for b in ubicaciones]
            idx_mayor = areas.index(max(areas))
            personaje = identificar_personaje(encodings[idx_mayor])
            cara_frac = areas[idx_mayor] / (alto_panel * ancho_panel)
            ultimo_personaje = personaje

        guion.append({
            "panel": p,
            "texto": texto if texto else "...",
            "personaje": personaje,
            "cara_frac": round(cara_frac, 3),
        })
    return guion


# ──────────────────────────────────────────────────────────────────────────
# Paso 4: voces naturales (XTTS-v2)
# ──────────────────────────────────────────────────────────────────────────
def _cargar_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dispositivo)
    return _tts


def generar_audios(guion):
    tts = _cargar_tts()
    personajes = sorted({l["personaje"] for l in guion})
    voces = {pid: tts.speakers[i % len(tts.speakers)] for i, pid in enumerate(personajes)}

    carpeta_audio = os.path.join(CARPETA_TRABAJO, "audio")
    os.makedirs(carpeta_audio, exist_ok=True)
    for i, linea in enumerate(guion):
        print(f"[4/6] Generando voz {i+1}/{len(guion)}...", flush=True)
        ruta_audio = os.path.join(carpeta_audio, f"linea_{i+1:02d}.wav")
        tts.tts_to_file(text=linea["texto"], speaker=voces[linea["personaje"]], language="es", file_path=ruta_audio)
        linea["audio"] = ruta_audio
    return guion


# ──────────────────────────────────────────────────────────────────────────
# Paso 4b: capa generativa — rostro con audio (SadTalker) + escena (AnimateDiff)
#
# Estos modelos son mucho más pesados/frágiles que el resto del pipeline.
# Por eso cada llamada está protegida con try/except: si algo falla en un
# panel puntual, ESE panel cae de vuelta al Ken Burns (siempre confiable)
# en vez de tumbar el video completo.
#
# NOTA IMPORTANTE sobre el cambio de AnimateDiff a paralaje por profundidad:
# AnimateDiff (Stable Diffusion + IP-Adapter) no hace una animación 'ligera' —
# REGENERA la imagen desde cero guiándose vagamente en el original. No importa
# qué tan bajo se ponga el parámetro de intensidad, el resultado siempre corre
# el riesgo de distorsionar el dibujo, porque así funciona un modelo de
# difusión: no es la herramienta correcta para 'mover un dibujo ya hecho sin
# tocarlo'. En su lugar, usamos paralaje basado en un mapa de profundidad
# (MiDaS, open source): se estima qué está más cerca/lejos en la imagen y se
# desplaza cada parte a distinta velocidad, simulando una cámara en movimiento.
# Esto NO regenera ni un solo píxel — solo reacomoda posiciones — así que es
# matemáticamente imposible que distorsione el arte.
# ──────────────────────────────────────────────────────────────────────────
UMBRAL_PRIMER_PLANO = 0.12       # rostro ocupa >12% del panel → boca con audio

_sadtalker_listo = False
_modelo_profundidad = None
_transform_profundidad = None


def decidir_modo_animacion(guion):
    config = {}
    for linea in guion:
        nombre = os.path.basename(linea["panel"])
        if linea.get("cara_frac", 0) >= UMBRAL_PRIMER_PLANO:
            config[nombre] = "rostro_audio"
        else:
            config[nombre] = "escena"
    return config


def _preparar_sadtalker():
    global _sadtalker_listo
    if _sadtalker_listo:
        return
    if not os.path.isdir("SadTalker"):
        subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"], check=True)
    subprocess.run(["pip", "install", "-q", "-r", "SadTalker/requirements.txt"], check=True)
    subprocess.run(["bash", "SadTalker/scripts/download_models.sh"], check=True, cwd="SadTalker")
    _sadtalker_listo = True


def animar_rostro_con_audio(ruta_imagen, ruta_audio, nombre_salida):
    """Devuelve la ruta del video con boca sincronizada, o None si falla."""
    try:
        _preparar_sadtalker()
        carpeta_temp = os.path.join(CARPETA_TRABAJO, f"_sadtalker_{nombre_salida}")
        os.makedirs(carpeta_temp, exist_ok=True)
        subprocess.run([
            "python", "SadTalker/inference.py",
            "--driven_audio", os.path.abspath(ruta_audio),
            "--source_image", os.path.abspath(ruta_imagen),
            "--result_dir", os.path.abspath(carpeta_temp),
            "--still", "--preprocess", "full",
        ], check=True, cwd=".")
        import glob as _glob
        candidatos = _glob.glob(os.path.join(carpeta_temp, "**", "*.mp4"), recursive=True)
        if not candidatos:
            return None
        destino = os.path.join(CARPETA_TRABAJO, f"{nombre_salida}_rostro.mp4")
        shutil.copy(candidatos[0], destino)
        return destino
    except Exception as e:
        print(f"⚠️ SadTalker falló en {nombre_salida}, se usará Ken Burns en su lugar. Detalle: {e}")
        return None


def _cargar_modelo_profundidad():
    global _modelo_profundidad, _transform_profundidad
    if _modelo_profundidad is not None:
        return _modelo_profundidad, _transform_profundidad
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    modelo.eval().to(dispositivo)
    transformador = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    _modelo_profundidad, _transform_profundidad = modelo, transformador
    return modelo, transformador


def _mapa_profundidad(imagen_pil):
    modelo, transform = _cargar_modelo_profundidad()
    dispositivo = next(modelo.parameters()).device
    arr = np.array(imagen_pil.convert("RGB"))
    entrada = transform(arr).to(dispositivo)
    with torch.no_grad():
        prediccion = modelo(entrada)
        prediccion = torch.nn.functional.interpolate(
            prediccion.unsqueeze(1), size=arr.shape[:2], mode="bicubic", align_corners=False,
        ).squeeze()
    profundidad = prediccion.cpu().numpy()
    profundidad = (profundidad - profundidad.min()) / (profundidad.max() - profundidad.min() + 1e-8)
    return profundidad  # 0..1, más alto = más cerca de la cámara


def animar_escena(ruta_imagen, nombre_salida, duracion=3.0, fps=24, amplitud_px=22):
    """Movimiento de cámara con paralaje real (profundidad estimada) — NO
    regenera el dibujo, solo desplaza posiciones. Devuelve la ruta del clip,
    o None si falla."""
    try:
        img_pil = Image.open(ruta_imagen).convert("RGB")
        profundidad = _mapa_profundidad(img_pil)
        arr = np.array(img_pil)
        h, w = arr.shape[:2]
        n_frames = max(int(duracion * fps), 2)

        map_x_base, map_y_base = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        desplazamiento_rel = ((profundidad - 0.5) * 2).astype(np.float32)  # -1..1: negativo=fondo, positivo=primer plano

        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            dx = np.float32(amplitud_px * np.sin(t * np.pi))  # barrido suave de ida
            map_x = (map_x_base + (dx * desplazamiento_rel)).astype(np.float32)
            frame = cv2.remap(arr, map_x, map_y_base, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frames.append(frame)

        from moviepy.editor import ImageSequenceClip
        destino = os.path.join(CARPETA_TRABAJO, f"{nombre_salida}_escena.mp4")
        ImageSequenceClip(frames, fps=fps).write_videofile(destino, codec="libx264", audio=False, logger=None)
        return destino
    except Exception as e:
        print(f"⚠️ Paralaje de profundidad falló en {nombre_salida}, se usará Ken Burns en su lugar. Detalle: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# Paso 5: animación base (Ken Burns) + ensamblado
# ──────────────────────────────────────────────────────────────────────────
def cubrir_lienzo(imagen_pil, ancho, alto):
    """Escala y recorta al centro para LLENAR el cuadro completo, sin bordes
    negros — a diferencia de 'contain', esto recorta el sobrante en vez de
    dejar espacio vacío."""
    w, h = imagen_pil.size
    escala = max(ancho / w, alto / h)
    nuevo_w, nuevo_h = round(w * escala), round(h * escala)
    imagen_escalada = imagen_pil.resize((nuevo_w, nuevo_h), Image.LANCZOS)
    x0 = (nuevo_w - ancho) // 2
    y0 = (nuevo_h - alto) // 2
    return imagen_escalada.crop((x0, y0, x0 + ancho, y0 + alto))


def cubrir_lienzo_clip(clip, ancho, alto):
    """Igual que cubrir_lienzo pero para clips de video (SadTalker/AnimateDiff
    no vienen en el tamaño del lienzo final, hay que ajustarlos)."""
    w, h = clip.size
    escala = max(ancho / w, alto / h)
    clip = clip.resize(escala)
    return clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=ancho, height=alto)


def ensamblar_video(guion, ruta_salida, zoom_final=1.18, usar_generativa=True):
    from moviepy.editor import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips

    def clip_ken_burns(ruta_imagen, duracion):
        img = Image.open(ruta_imagen).convert("RGB")
        img = cubrir_lienzo(img, ANCHO_BASE, ALTO_BASE)
        ruta_temp = ruta_imagen.replace(".png", "_lienzo.png")
        img.save(ruta_temp)
        clip = ImageClip(ruta_temp).set_duration(duracion)
        clip = clip.resize(lambda t: 1 + (zoom_final - 1) * (t / duracion))
        return clip.set_position(("center", "center"))

    config_animacion = decidir_modo_animacion(guion) if usar_generativa else {}

    clips = []
    for i, linea in enumerate(guion):
        print(f"[5/6] Animando panel {i+1}/{len(guion)} (modo: {config_animacion.get(os.path.basename(linea['panel']), 'ken_burns')})...", flush=True)
        audio = AudioFileClip(linea["audio"])
        duracion = max(audio.duration, 1.5)
        nombre = os.path.basename(linea["panel"])
        modo = config_animacion.get(nombre)
        clip_final = None

        if modo == "rostro_audio":
            ruta_video = animar_rostro_con_audio(linea["panel"], linea["audio"], f"linea_{i+1:02d}")
            if ruta_video:
                vc = VideoFileClip(ruta_video)
                vc = vc.set_duration(min(duracion + 0.3, vc.duration))
                clip_final = cubrir_lienzo_clip(vc, ANCHO_BASE, ALTO_BASE)
        elif modo == "escena":
            ruta_video = animar_escena(linea["panel"], f"linea_{i+1:02d}", duracion=duracion)
            if ruta_video:
                vc = cubrir_lienzo_clip(VideoFileClip(ruta_video), ANCHO_BASE, ALTO_BASE)
                clip_final = vc.set_duration(duracion).without_audio().set_audio(audio)

        if clip_final is None:  # respaldo: Ken Burns si no hubo generativa o si falló
            clip_final = clip_ken_burns(linea["panel"], duracion).set_audio(audio)

        clips.append(clip_final)

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(ruta_salida, fps=24, codec="libx264", audio_codec="aac", logger=None)
    return ruta_salida


# ──────────────────────────────────────────────────────────────────────────
# Paso 6: escalado final a 4K real
#
# El paso 1 (preprocesamiento) ya aplicó Real-ESRGAN x4 a cada panel — el video
# base resultante ya viene en una resolución igual o mayor a 4K. Volver a correr
# la IA de escalado por cada frame del video sería trabajo redundante y
# extremadamente pesado (esto fue justo lo que colapsó la sesión de Colab).
# Aquí solo se ajusta al lienzo exacto 3840x2160 con FFmpeg (rápido, sin GPU).
# ──────────────────────────────────────────────────────────────────────────
def escalar_video_a_4k(ruta_video_entrada, ruta_video_salida):
    subprocess.run([
        "ffmpeg", "-y", "-i", ruta_video_entrada,
        "-vf", f"scale={ANCHO_4K}:{ALTO_4K}:flags=lanczos,setsar=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        "-c:a", "copy",
        ruta_video_salida,
    ], check=True)
    return ruta_video_salida


# ──────────────────────────────────────────────────────────────────────────
# Pipeline completo — esto es lo único que la interfaz llama
# ──────────────────────────────────────────────────────────────────────────
def generar_video(ruta_imagen_subida, progress=gr.Progress()):
    if ruta_imagen_subida is None:
        raise gr.Error("Sube una página de cómic primero.")

    shutil.rmtree(CARPETA_TRABAJO, ignore_errors=True)
    os.makedirs(CARPETA_TRABAJO, exist_ok=True)

    progress(0.05, desc="Detectando y recortando paneles...")
    ruta_pagina = os.path.join(CARPETA_TRABAJO, "pagina_original.png")
    Image.open(ruta_imagen_subida).convert("RGB").save(ruta_pagina)
    # Importante: se detectan los paneles ANTES de escalar/avivar color —
    # el escalado altera los márgenes blancos que separan viñetas y hace
    # fallar la detección si se aplica primero.
    paneles_originales = detectar_paneles(ruta_pagina)
    print(f"[1/6] {len(paneles_originales)} paneles detectados.", flush=True)

    progress(0.20, desc="Más resolución y color más vivo por panel...")
    paneles = []
    for i, p in enumerate(paneles_originales):
        print(f"[2/6] Procesando panel {i+1}/{len(paneles_originales)}...", flush=True)
        paneles.append(preprocesar_imagen(p))
    print("[2/6] Preprocesamiento de paneles completo.", flush=True)

    progress(0.35, desc="Leyendo diálogos y reconociendo personajes...")
    guion = generar_guion(paneles)
    print(f"[3/6] Guion generado ({len(guion)} líneas).", flush=True)

    progress(0.55, desc="Generando voces naturales...")
    guion = generar_audios(guion)
    print("[4/6] Voces generadas.", flush=True)

    progress(0.75, desc="Animando y ensamblando video...")
    video_base = ensamblar_video(guion, os.path.join(CARPETA_TRABAJO, "video_base.mp4"))
    print("[5/6] Video base ensamblado.", flush=True)

    progress(0.90, desc="Escalando a 4K real...")
    video_4k = escalar_video_a_4k(video_base, os.path.join(CARPETA_TRABAJO, "video_4K.mp4"))
    print("[6/6] Escalado a 4K completo. ¡Listo!", flush=True)

    progress(1.0, desc="¡Listo!")
    return video_4k


# ──────────────────────────────────────────────────────────────────────────
# Interfaz — muy simple: subir imagen + botón, nada más
# ──────────────────────────────────────────────────────────────────────────
interfaz = gr.Interface(
    fn=generar_video,
    inputs=gr.Image(type="filepath", label="Sube tu página de cómic"),
    outputs=gr.Video(label="Video generado (4K)"),
    title="📖➡️🎬 Cómic Animado",
    description=(
        "Sube una página de cómic/webcómic y obtén un video narrado con voces naturales, "
        "en 4K. Todo corre localmente con herramientas open source (puede tardar varios "
        "minutos, sobre todo la primera vez que descarga los modelos)."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    interfaz.launch(share=True)  # share=True crea un link público temporal (útil fuera de Colab también)
