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
import glob
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
CARPETA_ESRGAN = "realesrgan_ncnn"
MODELO_ESRGAN = "realesrgan-x4plus-anime"
ANCHO_4K, ALTO_4K = 3840, 2160

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # evita bloqueo por licencia de XTTS-v2

_tts = None            # se carga una sola vez, la primera vez que se usa
_ocr_reader = None
_binario_esrgan = None


# ──────────────────────────────────────────────────────────────────────────
# Paso 0: preparar Real-ESRGAN (ejecutable precompilado, sin compilar nada)
# ──────────────────────────────────────────────────────────────────────────
def preparar_realesrgan():
    global _binario_esrgan
    if _binario_esrgan and os.path.exists(_binario_esrgan):
        return _binario_esrgan

    candidatos = glob.glob(f"{CARPETA_ESRGAN}/**/realesrgan-ncnn-vulkan", recursive=True)
    if candidatos:
        _binario_esrgan = candidatos[0]
        os.chmod(_binario_esrgan, 0o755)
        return _binario_esrgan

    os.makedirs(CARPETA_ESRGAN, exist_ok=True)
    zip_path = os.path.join(CARPETA_ESRGAN, "realesrgan_ncnn.zip")
    url = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    )
    subprocess.run(["wget", "-nc", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-o", zip_path, "-d", CARPETA_ESRGAN], check=True)

    candidatos = glob.glob(f"{CARPETA_ESRGAN}/**/realesrgan-ncnn-vulkan", recursive=True)
    if not candidatos:
        raise FileNotFoundError(
            "No se pudo preparar Real-ESRGAN. Si estás fuera de Linux (Windows/Mac), "
            "necesitas el ejecutable equivalente para tu sistema operativo desde: "
            "https://github.com/xinntao/Real-ESRGAN/releases"
        )
    _binario_esrgan = candidatos[0]
    os.chmod(_binario_esrgan, 0o755)
    return _binario_esrgan


def escalar_con_esrgan(ruta_entrada, ruta_salida, factor=2):
    binario = preparar_realesrgan()
    subprocess.run(
        [binario, "-i", ruta_entrada, "-o", ruta_salida, "-n", MODELO_ESRGAN, "-s", str(factor)],
        check=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# Paso 1: preprocesamiento — más resolución + color más vivo (sin alterar nada)
# ──────────────────────────────────────────────────────────────────────────
def preprocesar_imagen(ruta_entrada, factor_saturacion=1.15, limite_lado_px=2400):
    img_pil_orig = Image.open(ruta_entrada).convert("RGB")
    base, _ = os.path.splitext(ruta_entrada)

    if max(img_pil_orig.size) < limite_lado_px:
        ruta_escalada = f"{base}_escalada.png"
        escalar_con_esrgan(ruta_entrada, ruta_escalada, factor=2)
        img_pil = Image.open(ruta_escalada).convert("RGB")
    else:
        img_pil = img_pil_orig

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
# ──────────────────────────────────────────────────────────────────────────
def escalar_video_a_4k(ruta_video_entrada, ruta_video_salida):
    binario = preparar_realesrgan()
    carpeta_in = os.path.join(CARPETA_TRABAJO, "frames_originales")
    carpeta_esrgan = os.path.join(CARPETA_TRABAJO, "frames_esrgan")
    carpeta_out = os.path.join(CARPETA_TRABAJO, "frames_4k")
    for carpeta in (carpeta_in, carpeta_esrgan, carpeta_out):
        shutil.rmtree(carpeta, ignore_errors=True)
        os.makedirs(carpeta, exist_ok=True)

    subprocess.run(["ffmpeg", "-y", "-i", ruta_video_entrada, f"{carpeta_in}/f_%06d.png"], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", ruta_video_entrada, os.path.join(CARPETA_TRABAJO, "audio_original.aac")], check=True)

    subprocess.run([binario, "-i", carpeta_in, "-o", carpeta_esrgan, "-n", MODELO_ESRGAN, "-s", "4"], check=True)

    for nombre in sorted(os.listdir(carpeta_esrgan)):
        img = cv2.imread(os.path.join(carpeta_esrgan, nombre))
        img = cv2.resize(img, (ANCHO_4K, ALTO_4K), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(carpeta_out, nombre), img)

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24", "-i", f"{carpeta_out}/f_%06d.png",
        "-i", os.path.join(CARPETA_TRABAJO, "audio_original.aac"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        "-c:a", "aac", "-shortest", ruta_video_salida,
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
