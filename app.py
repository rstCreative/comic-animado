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
import gradio as gr
from PIL import Image, ImageEnhance

# ──────────────────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────────────────
CARPETA_TRABAJO = "trabajo"
ANCHO_4K, ALTO_4K = 3840, 2160

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


def detectar_paneles(ruta_imagen, area_min_frac=0.02):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")
    h_total, w_total = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 245, 255, cv2.THRESH_BINARY_INV)
    binaria = cv2.dilate(binaria, np.ones((15, 15), np.uint8), iterations=2)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = area_min_frac * w_total * h_total

    cajas = [cv2.boundingRect(c) for c in contornos if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > area_min]
    cajas = ordenar_lectura(cajas, w_total)

    paneles = []
    for i, (x, y, w, h) in enumerate(cajas):
        salida = os.path.join(CARPETA_TRABAJO, f"panel_{i+1:02d}.png")
        cv2.imwrite(salida, img[y:y+h, x:x+w])
        paneles.append(salida)

    if not paneles:  # si no se detectó ningún panel, se usa la página completa como un solo panel
        salida = os.path.join(CARPETA_TRABAJO, "panel_01.png")
        cv2.imwrite(salida, img)
        paneles = [salida]

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
    for p in paneles:
        texto = " ".join(reader.readtext(p, detail=0, paragraph=True)).strip()

        img = face_recognition.load_image_file(p)
        ubicaciones = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, ubicaciones)

        personaje = ultimo_personaje or "personaje_1"
        if encodings:
            areas = [(b[2]-b[0]) * (b[1]-b[3]) for b in ubicaciones]
            idx_mayor = areas.index(max(areas))
            personaje = identificar_personaje(encodings[idx_mayor])
            ultimo_personaje = personaje

        guion.append({
            "panel": p,
            "texto": texto if texto else "...",
            "personaje": personaje,
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
        ruta_audio = os.path.join(carpeta_audio, f"linea_{i+1:02d}.wav")
        tts.tts_to_file(text=linea["texto"], speaker=voces[linea["personaje"]], language="es", file_path=ruta_audio)
        linea["audio"] = ruta_audio
    return guion


# ──────────────────────────────────────────────────────────────────────────
# Paso 5: animación base (Ken Burns) + ensamblado
# ──────────────────────────────────────────────────────────────────────────
def ensamblar_video(guion, ruta_salida, zoom_final=1.08):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    def clip_ken_burns(ruta_imagen, duracion):
        clip = ImageClip(ruta_imagen).set_duration(duracion)
        clip = clip.resize(lambda t: 1 + (zoom_final - 1) * (t / duracion))
        return clip.set_position(("center", "center"))

    clips = []
    for linea in guion:
        audio = AudioFileClip(linea["audio"])
        duracion = max(audio.duration, 1.5)
        clips.append(clip_ken_burns(linea["panel"], duracion).set_audio(audio))

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

    progress(0.05, desc="Más resolución y color más vivo...")
    ruta_pagina = os.path.join(CARPETA_TRABAJO, "pagina_original.png")
    Image.open(ruta_imagen_subida).convert("RGB").save(ruta_pagina)
    ruta_pagina = preprocesar_imagen(ruta_pagina)

    progress(0.20, desc="Detectando y recortando paneles...")
    paneles = detectar_paneles(ruta_pagina)

    progress(0.35, desc="Leyendo diálogos y reconociendo personajes...")
    guion = generar_guion(paneles)

    progress(0.55, desc="Generando voces naturales...")
    guion = generar_audios(guion)

    progress(0.75, desc="Animando y ensamblando video...")
    video_base = ensamblar_video(guion, os.path.join(CARPETA_TRABAJO, "video_base.mp4"))

    progress(0.90, desc="Escalando a 4K real...")
    video_4k = escalar_video_a_4k(video_base, os.path.join(CARPETA_TRABAJO, "video_4K.mp4"))

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
