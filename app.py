"""
MÓDULO 1 — Generador de Personajes Manga/Anime (Qwen-Image + LoRA)
===================================================================
Primer módulo del pipeline (por partes, como acordamos). Este módulo NO
depende de nada de "Cómic Animado" — es standalone: entrena un LoRA por
personaje y genera stickers/imágenes controlando atributos por interfaz.

Pensado 100% para Google Colab (GPU T4/A100).

Flujo:
  1. Subís 5-15 fotos/dibujos del personaje en DISTINTAS POSTURAS
     (de pie, sentado, de perfil, gesto de mano, etc. — la variedad de
     postura en el dataset es justamente lo que le enseña al LoRA a
     dibujar al personaje en poses nuevas más adelante).
  2. Entrenás el LoRA una vez (dura 15-40 min según steps).
  3. Generás imágenes eligiendo atributos por dropdown: color de ojos,
     color/largo/estilo de pelo, accesorios, ropa, calzado, edad
     (adulto/adolescente/niño/bebé), y una descripción de pose.

NOTA DE DISEÑO (decisión que tomamos juntos):
- El control de pose es "por texto + variedad en el dataset de
  entrenamiento", NO por esqueleto de referencia (ControlNet). Es más
  simple de armar y de correr en Colab; a cambio, la pose exacta no
  está garantizada al 100%, depende de qué tan bien el LoRA generalizó.
- Para cambiar UN atributo (ej: sólo el color de ojos) el modo elegido
  es "regenerar la imagen completa con el mismo seed + prompt modificado"
  en vez de inpainting local. Es más simple de programar y de correr en
  Colab (sin necesidad de máscaras), a cambio de que el resto del cuerpo
  puede variar levemente entre una generación y otra. Si más adelante
  hace falta precisión pixel-perfect en un solo atributo, ese sería un
  Módulo 2 aparte (inpainting con máscara).

Uso en Colab:
    !pip install -q -r requirements_generador.txt
    !python generador_personajes_qwen.py
"""

import os
import subprocess
import random
import uuid

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────────────────
MODELO_BASE = "Qwen/Qwen-Image"
CARPETA_DIFFUSERS = "diffusers"  # clon del repo, donde vive el script de entrenamiento
CARPETA_DATASETS = "personajes_dataset"   # personajes_dataset/<nombre>/*.png
CARPETA_LORAS = "personajes_lora"         # personajes_lora/<nombre>/pytorch_lora_weights.safetensors
CARPETA_SALIDAS = "salidas_generador"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
os.makedirs(CARPETA_DATASETS, exist_ok=True)
os.makedirs(CARPETA_LORAS, exist_ok=True)
os.makedirs(CARPETA_SALIDAS, exist_ok=True)

_pipe_actual = None          # pipeline cargado (se recarga si cambia el LoRA activo)
_lora_actual_cargado = None  # nombre del personaje cuyo LoRA está cargado ahora


# ──────────────────────────────────────────────────────────────────────────
# PASO 1 — Preparar el dataset de un personaje a partir de imágenes subidas
# ──────────────────────────────────────────────────────────────────────────
def _detectar_paneles_simple(ruta_imagen, area_min_frac=0.01):
    """Separa una hoja/grilla de referencia en paneles individuales, por
    contorno contra el fondo (misma idea validada en preparar_dataset_grillas.py,
    sin dilatar — acá los paneles suelen estar pegados). Si no encuentra más
    de 1 región, asume que la imagen entera es un solo panel."""
    img = cv2.imread(ruta_imagen)
    h, w = img.shape[:2]
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_fondo = int(np.median([gris[0, 0], gris[0, -1], gris[-1, 0], gris[-1, -1]]))
    _, binaria = cv2.threshold(gris, max(color_fondo - 8, 0), 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = area_min_frac * w * h
    cajas = [cv2.boundingRect(c) for c in contornos]
    cajas = [c for c in cajas if c[2] * c[3] > area_min]

    if len(cajas) <= 1:
        return [(0, 0, w, h)]

    cajas = sorted(cajas, key=lambda c: c[1])
    filas = []
    for c in cajas:
        colocada = False
        for fila in filas:
            if abs(c[1] - fila[0][1]) < 0.06 * w:
                fila.append(c)
                colocada = True
                break
        if not colocada:
            filas.append([c])
    resultado = []
    for fila in filas:
        resultado.extend(sorted(fila, key=lambda c: c[0]))
    return resultado


def agregar_al_dataset(nombre_personaje, archivos_subidos, descripcion_base, es_grilla, progress=gr.Progress()):
    """Agrega imágenes al dataset de un personaje de forma INCREMENTAL —
    podés llamarla varias veces con distintas tandas de fotos, no pisa lo
    que ya había. Si 'es_grilla' está activo, cada imagen subida se separa
    primero en paneles individuales (hoja de referencia con varias poses).
    Cada panel resultante pasa por el mismo control de calidad del
    generador (manos/cabezas/piernas de más) antes de guardarse: lo que
    pasa va a personajes_dataset/<nombre>/, lo sospechoso va a
    personajes_dataset/revisar_manual/<nombre>/ para que lo mires vos.
    Devuelve un string-resumen para mostrar en la interfaz."""
    if not nombre_personaje or not nombre_personaje.strip():
        raise gr.Error("Ponele un nombre corto al personaje (ej: 'ren', 'kai').")
    if not archivos_subidos:
        raise gr.Error("Subí al menos una imagen.")

    nombre_personaje = nombre_personaje.strip().lower().replace(" ", "_")
    token = f"{nombre_personaje}_chr"
    carpeta_personaje = os.path.join(CARPETA_DATASETS, nombre_personaje)
    carpeta_revisar = os.path.join(CARPETA_DATASETS, "revisar_manual", nombre_personaje)
    os.makedirs(carpeta_personaje, exist_ok=True)
    os.makedirs(carpeta_revisar, exist_ok=True)

    agregadas, a_revisar, total_paneles = 0, 0, 0

    for idx_archivo, ruta_subida in enumerate(archivos_subidos):
        progress(idx_archivo / len(archivos_subidos), desc=f"Procesando archivo {idx_archivo+1}/{len(archivos_subidos)}...")
        cajas = _detectar_paneles_simple(ruta_subida) if es_grilla else None
        img_original = cv2.imread(ruta_subida)

        if cajas is None:
            recortes = [Image.open(ruta_subida).convert("RGB")]
        else:
            recortes = []
            for (x, y, w, h) in cajas:
                m = 2  # sacar un pelo de borde
                panel = img_original[y+m:y+h-m, x+m:x+w-m]
                if panel.size == 0:
                    continue
                recortes.append(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
        total_paneles += len(recortes)

        for img in recortes:
            sufijo = uuid.uuid4().hex[:8]  # nombre único: subidas incrementales no se pisan entre sí
            _, motivos, _ = revisar_calidad_imagen(img, quitar_fondo(img))
            destino_carpeta = carpeta_personaje if not motivos else carpeta_revisar
            nombre_archivo = f"{nombre_personaje}_{sufijo}.png"
            img.save(os.path.join(destino_carpeta, nombre_archivo))
            with open(os.path.join(destino_carpeta, nombre_archivo.replace(".png", ".txt")), "w") as f:
                f.write(f"{token}, {descripcion_base}")
            if motivos:
                a_revisar += 1
                print(f"⚠️ {nombre_archivo}: {', '.join(motivos)} → revisar_manual/", flush=True)
            else:
                agregadas += 1

    progress(1.0, desc="Listo.")
    n_limpias = len([f for f in os.listdir(carpeta_personaje) if f.endswith(".png")])
    resumen = (
        f"✅ {agregadas} imágenes agregadas a '{nombre_personaje}' "
        f"({'recortadas de ' + str(len(archivos_subidos)) + ' hoja(s), ' + str(total_paneles) + ' paneles' if es_grilla else str(len(archivos_subidos)) + ' archivo(s)'}).\n"
    )
    if a_revisar:
        resumen += f"⚠️ {a_revisar} quedaron en revisar_manual/{nombre_personaje}/ por posibles manos/cabezas/piernas de más.\n"
    resumen += f"📊 Total acumulado en el dataset de '{nombre_personaje}': {n_limpias} imágenes."
    return resumen


# ──────────────────────────────────────────────────────────────────────────
# PASO 2 — Entrenar el LoRA (script oficial de diffusers para Qwen-Image)
# ──────────────────────────────────────────────────────────────────────────
def _asegurar_script_entrenamiento():
    """Clona diffusers si hace falta y confirma que el script de Qwen-Image
    dreambooth-lora existe (vive en examples/dreambooth/ en el repo oficial)."""
    if not os.path.isdir(CARPETA_DIFFUSERS):
        subprocess.run(["git", "clone", "--depth", "1",
                         "https://github.com/huggingface/diffusers.git"], check=True)
    ruta_script = os.path.join(CARPETA_DIFFUSERS, "examples", "dreambooth",
                                "train_dreambooth_lora_qwen_image.py")
    if not os.path.exists(ruta_script):
        raise RuntimeError(
            "No se encontró train_dreambooth_lora_qwen_image.py en el clon de diffusers. "
            "El script pudo haber sido renombrado río arriba; revisar "
            "examples/dreambooth/README_qwen.md en el repo de diffusers."
        )
    return ruta_script


def entrenar_lora_personaje(nombre_personaje, carpeta_dataset, token_identificador,
                             pasos=1200, rank=32, resolucion=1024, progress=gr.Progress()):
    """Lanza el entrenamiento LoRA vía 'accelerate launch'. Devuelve la ruta
    de la carpeta con los pesos entrenados."""
    ruta_script = _asegurar_script_entrenamiento()
    carpeta_salida_lora = os.path.join(CARPETA_LORAS, nombre_personaje)
    os.makedirs(carpeta_salida_lora, exist_ok=True)

    progress(0.05, desc=f"Entrenando LoRA de {nombre_personaje} ({pasos} pasos, ~{pasos // 40} min estimados)...")
    comando = [
        "accelerate", "launch", ruta_script,
        f"--pretrained_model_name_or_path={MODELO_BASE}",
        f"--instance_data_dir={carpeta_dataset}",
        f"--output_dir={carpeta_salida_lora}",
        f"--instance_prompt={token_identificador}",
        f"--resolution={resolucion}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={pasos}",
        f"--rank={rank}",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--seed=0",
    ]
    print("[Entrenamiento] Ejecutando:", " ".join(comando), flush=True)
    subprocess.run(comando, check=True)

    progress(1.0, desc="LoRA entrenado.")
    return carpeta_salida_lora


# ──────────────────────────────────────────────────────────────────────────
# PASO 3 — Cargar el pipeline de inferencia con el LoRA activo
# ──────────────────────────────────────────────────────────────────────────
def _cargar_pipeline(nombre_personaje):
    global _pipe_actual, _lora_actual_cargado
    if _pipe_actual is not None and _lora_actual_cargado == nombre_personaje:
        return _pipe_actual  # ya está cargado el LoRA correcto, no repetir trabajo

    from diffusers import QwenImagePipeline

    if _pipe_actual is None:
        print("[Inferencia] Cargando Qwen-Image base (una sola vez)...", flush=True)
        _pipe_actual = QwenImagePipeline.from_pretrained(MODELO_BASE, torch_dtype=DTYPE)
        _pipe_actual.enable_model_cpu_offload()
    else:
        _pipe_actual.unload_lora_weights()  # sacar el LoRA anterior antes de poner el nuevo

    ruta_lora = os.path.join(CARPETA_LORAS, nombre_personaje)
    if not os.path.isdir(ruta_lora):
        raise gr.Error(f"No hay un LoRA entrenado para '{nombre_personaje}' todavía.")
    print(f"[Inferencia] Cargando LoRA de {nombre_personaje}...", flush=True)
    _pipe_actual.load_lora_weights(ruta_lora)
    _lora_actual_cargado = nombre_personaje
    return _pipe_actual


def _cargar_pipeline_sin_lora():
    """Para el modo 'Generación libre' — el modelo base de Qwen-Image sin
    ningún LoRA de personaje cargado. Reusa la misma instancia del modelo
    (evita duplicar VRAM); si había un LoRA cargado, lo saca primero."""
    global _pipe_actual, _lora_actual_cargado
    from diffusers import QwenImagePipeline

    if _pipe_actual is None:
        print("[Inferencia] Cargando Qwen-Image base (una sola vez)...", flush=True)
        _pipe_actual = QwenImagePipeline.from_pretrained(MODELO_BASE, torch_dtype=DTYPE)
        _pipe_actual.enable_model_cpu_offload()
    elif _lora_actual_cargado is not None:
        print("[Inferencia] Sacando el LoRA de personaje para generación libre...", flush=True)
        _pipe_actual.unload_lora_weights()
        _lora_actual_cargado = None
    return _pipe_actual


# ──────────────────────────────────────────────────────────────────────────
# PASO 4 — Banco de atributos (etiqueta en español → fragmento de prompt)
# Cubre lo que pediste: ojos, pelo (color/largo/estilo), accesorios, ropa
# (incl. moda masculina), calzado, edad (incl. niños/bebés), pose.
# ──────────────────────────────────────────────────────────────────────────
COLOR_OJOS = {
    "Marrón": "brown eyes", "Azul": "blue eyes", "Verde": "green eyes",
    "Gris": "gray eyes", "Ámbar": "amber eyes", "Violeta": "violet eyes",
    "Heterocromía (dos colores)": "heterochromia, one blue one brown eye",
}
TIPO_OJOS = {
    "Grandes clásicos": "classic large round anime eyes",
    "Almendrados": "almond-shaped eyes",
    "Redondos": "round doll-like eyes",
    "Rasgados": "sharp narrow eyes",
    "Tipo shōjo": "shoujo-style eyes with detailed sparkling highlights",
    "Con efecto brillo": "eyes with glossy sparkle highlight effect",
    "Serios": "sharp serious eyes",
    "Anime chibi": "simplified chibi-style eyes",
    "Enojados": "angry narrowed eyes",
}

COLOR_PELO = {
    "Negro": "black hair", "Castaño": "brown hair", "Rubio": "blonde hair",
    "Pelirrojo": "red hair", "Blanco/plateado": "silver white hair",
    "Azul": "blue hair", "Rosa": "pink hair", "Verde": "green hair",
    "Morado": "purple hair", "Multicolor": "multicolored gradient hair",
}
LARGO_PELO = {
    "Corto": "short hair", "Mediano": "medium length hair",
    "Largo": "long hair", "Muy largo": "very long hair",
}
ESTILO_PELO = {
    "Liso": "straight", "Ondulado": "wavy", "Rizado": "curly",
    "Coleta": "ponytail", "Trenza": "braided", "Moño": "hair bun",
    "Despeinado": "messy",
}

ACCESORIOS = {
    "Ninguno": "", "Lentes": "glasses", "Gafas de sol": "sunglasses",
    "Aretes": "earrings", "Collar": "necklace", "Gorra": "cap", "Sombrero": "hat",
    "Diadema": "headband", "Bufanda": "scarf", "Guantes": "gloves",
    "Reloj": "wristwatch", "Mochila": "backpack",
    "Gorro de Navidad": "santa hat", "Audífonos": "headphones",
    "Delantal de cocina": "kitchen apron",
}

TIPO_ROPA = {
    "Casual": "casual streetwear outfit", "Uniforme escolar": "school uniform",
    "Deportiva": "sportswear, athletic outfit", "Formal/elegante": "formal elegant outfit",
    "Streetwear urbano": "trendy urban streetwear, stylish fit",
    "Pijama": "pajamas", "Pijama con capucha": "hooded pajama onesie",
    "Abrigo de invierno": "winter coat outfit",
    "Falda corta + top": "short pleated skirt with crop top",
    "Falda de tenis + top": "tennis skirt with sporty top",
    "Top corto + short": "cropped top with shorts",
    "Top cruzado + falda": "wrap crop top with skirt",
    "Jeans y top": "jeans with a top", "Vestido casual (sundress)": "casual sundress",
    "Vestido de noche": "elegant evening dress", "Vestido de fiesta": "party dress",
    "Vestido de novia": "wedding dress", "Toga de graduación": "graduation gown and cap",
    "Disfraz cosplay anime": "anime cosplay costume",
    "Disfraz kigurumi (animal)": "animal kigurumi onesie costume",
    "Disfraz de maid": "maid costume",
    "Disfraz de policía": "police uniform costume",
    "Disfraz de superhéroe/heroína": "superhero costume",
    "Disfraz de bruja (Halloween)": "witch Halloween costume",
    "Ropa de yoga": "yoga outfit, leggings and sports top",
    "Ropa de gimnasio": "gym workout outfit",
    "Outfit de otoño (bufanda)": "autumn outfit with scarf",
    "Outfit de primavera (floral)": "spring floral outfit",
    # Vestido de baño: SOLO disponible con Edad = Adulto (ver ROPA_SOLO_ADULTO
    # más abajo). No incluyo variantes tipo tanga/g-string, ni la sección de
    # ropa interior/lencería de la hoja que mandaste: queda afuera del
    # vocabulario del generador, no es solo un tema de gusto — para un
    # selector de ropa compartido con el resto de las edades, ese tipo de
    # prenda no entra en la lista de opciones, punto.
    "Bikini clásico": "classic bikini swimsuit",
    "Bikini deportivo": "athletic sports bikini",
    "Bikini estampado": "patterned bikini swimsuit",
    "Traje de baño entero": "one-piece swimsuit",
    "Pareo de playa (sobre traje de baño)": "beach sarong cover-up over swimsuit",
    "Short/bañador de baño": "men's swim trunks",
}
# Prendas bloqueadas salvo que Edad == 'Adulto' — control de código, no solo
# de prompt: si esto se combinara con Niño/niña o Bebé el pedido se corta acá,
# no depende de que el modelo "entienda" el negativo.
ROPA_SOLO_ADULTO = {"Bikini clásico", "Bikini deportivo", "Bikini estampado",
                     "Traje de baño entero", "Pareo de playa (sobre traje de baño)",
                     "Short/bañador de baño"}

CALZADO = {
    "Zapatillas/tenis": "sneakers", "Zapatos formales": "formal dress shoes",
    "Sandalias/chanclas": "sandals, flip-flops", "Botas": "boots",
    "Tacones": "high heels", "Descalzo": "barefoot",
}

EDAD = {
    "Adulto": "adult", "Adolescente": "teenager",
    "Niño/niña": "young child, non-sexualized, fully clothed, innocent, wholesome family-friendly style",
    "Bebé": "baby toddler, non-sexualized, fully clothed, innocent, wholesome family-friendly style",
}

GENERO = {"Mujer": "female", "Hombre": "male", "No binario / ambiguo": "androgynous"}

# ── Mascota / animal de compañía (para escenas tipo 'con su perro') ──
MASCOTA = {
    "Ninguna": "",
    "Perro": "a dog", "Gato": "a cat", "Conejo": "a rabbit",
    "Hámster": "a hamster", "Tortuga": "a turtle",
    "Pez (pecera)": "a fish in a fish tank", "Ave (loro/canario)": "a pet bird",
    "Reptil": "a pet reptile", "Erizo": "a hedgehog",
}
COLLAR_MASCOTA = {
    "Ninguno": "",
    "Clásico": "wearing a classic collar", "De cuero": "wearing a leather collar",
    "Con nombre": "wearing a collar with a name tag",
    "Reflectivo": "wearing a reflective collar", "Con púas": "wearing a spiked collar",
    "Personalizado": "wearing a custom decorated collar",
    "Con cascabel": "wearing a collar with a bell",
}
ACTIVIDAD_MASCOTA = {
    "Sin especificar": "",
    "Jugando con la mascota": "playing with the pet",
    "Abrazando a la mascota": "hugging the pet",
    "Paseando a la mascota (correa)": "walking the pet on a leash",
    "Dando de comer a la mascota": "feeding the pet",
    "Durmiendo con la mascota": "sleeping next to the pet",
    "Cuidando a la mascota": "caring for the pet",
    "Selfie con la mascota": "taking a selfie with the pet",
    "Con la mascota en brazos": "holding the pet in her arms",
}

# ── Profesión/rol: paquete de vestuario + props implícitos por ocupación ──
PROFESION = {
    "Ninguna": "",
    "Doctora/Doctor": "doctor, white coat, stethoscope",
    "Cirujana/Cirujano": "surgeon, scrubs and surgical mask",
    "Ingeniera/Ingeniero": "engineer, hard hat and safety vest",
    "Azafata/Sobrecargo": "flight attendant uniform",
    "Camarera/Mesero": "waiter uniform, apron",
    "Policía": "police officer uniform",
    "Profesora/Profesor": "teacher, holding a book",
    "Diosa/Dios (fantasía)": "fantasy deity, elegant flowing gown, ethereal glowing light",
}

# ── Expresión facial, gesto de manos y ángulo de vista (turnaround) ──
# Tomado de las hojas de referencia que mandaste, dejando afuera todo lo que
# ya charlamos que no sumo (poses/prendas sexuales, lencería, etc.).
EXPRESION = {
    "Ninguna (uso el texto de pose libre)": "",
    "Feliz": "happy expression", "Triste": "sad expression",
    "Enojada": "angry expression", "Sorprendida": "surprised expression",
    "Tímida": "shy expression", "Avergonzada": "embarrassed blushing expression",
    "Riendo": "laughing expression", "Guiño": "winking",
    "Seria": "serious expression", "Pensativa": "thoughtful expression",
    "Molesta": "annoyed expression", "Llanto": "crying expression",
    "Dudosa": "unsure puzzled expression", "Miedosa": "scared fearful expression",
    "Enamorada": "expression in love with heart eyes",
    "Beso": "blowing a kiss expression",
    "Burlona": "teasing playful expression", "Emocionada": "excited thrilled expression",
    "Gritando": "screaming expression, mouth wide open",
    "Motivada": "motivated determined expression",
    "Sonrojada": "blushing shy expression", "Ojos cerrados (sonriendo)": "closed eyes, gentle smile expression",
}

GESTO = {
    "Ninguno": "",
    "Paz": "peace sign hand gesture", "Corazón con manos": "heart hand gesture",
    "Dedos en V": "V sign with fingers", "Señalar": "pointing at viewer gesture",
    "Pulgar arriba": "thumbs up gesture", "OK": "OK hand sign gesture",
    "Mano en la cara": "hand near face pose", "Saludo": "waving hello gesture",
    "Manos en la cintura": "hands on hips", "Pensando": "hand on chin thinking pose",
    "Agarrando su cabello": "hand touching own hair pose",
}

VISTA = {
    "Sin especificar": "",
    "Frente": "front view", "3/4 izquierda": "three-quarter view from left",
    "3/4 derecha": "three-quarter view from right",
    "Perfil izquierdo": "left side profile view", "Perfil derecho": "right side profile view",
    "Espalda": "back view", "Mirando atrás": "looking back over shoulder",
    "Vista desde arriba": "high angle view from above",
    "Vista desde abajo": "low angle view from below",
}

# ── Pose de cuerpo (postura general, no gesto de manos ni expresión) ──
POSE_CUERPO = {
    "Sin especificar": "",
    "Manos en bolsillo": "hands in pockets", "Cruzado de brazos": "arms crossed",
    "Apoyado en la pared": "leaning against a wall",
    "Sentado normal": "sitting normally", "Sentado con pierna cruzada": "sitting with legs crossed",
    "Apoyando el codo": "leaning on elbow", "Manos en mentón": "hand resting on chin",
    "Acostado boca arriba": "lying on back, relaxed, fully clothed",
    "Acostado de lado": "lying on side, relaxed, fully clothed",
    "Acostado boca abajo": "lying on stomach, relaxed, fully clothed",
    "Con almohada": "hugging a pillow, fully clothed",
    "Mirando hacia abajo": "looking down", "Mirando a un lado": "looking to the side",
    "Inclinarse hacia adelante": "leaning forward pose",
    "Manos en el cabello": "hands running through hair pose",
    "Mirando al horizonte": "looking towards the horizon",
    "Manos en la cabeza": "hands on head pose",
    "En el sofá": "sitting on a couch, relaxed",
    "Envuelta en una manta": "wrapped in a blanket, cozy pose",
    "Tomando café": "holding a cup of coffee",
    "Leyendo un libro": "reading a book",
    "Mirando TV": "watching TV, relaxed pose",
    "De compras": "carrying shopping bags",
    "Estirando": "stretching pose",
    "Escuchando música": "listening to music with headphones, relaxed",
    "Cocinando": "cooking in the kitchen",
    "Con snacks": "eating snacks, cozy pose",
    "Escribiendo mensajes": "texting on a phone",
    "Hablando por celular": "talking on the phone",
    "Probándose ropa": "trying on clothes in a fitting room",
    "Cantando con micrófono": "singing with a microphone",
    "Entrenando en el gimnasio": "working out at the gym",
    "Con taza caliente": "holding a hot drink mug",
    "Poniéndose los zapatos": "putting on shoes, tying laces",
    "Caminando en el parque": "walking in a park",
    "Sentada en un banco": "sitting on a park bench",
    "Con regalos de Navidad": "holding christmas gifts",
    "Senderismo (con mochila)": "hiking pose with a backpack",
    "Acampando": "camping pose, sitting by a tent",
    "Explorando": "exploring pose, looking around curiously",
    "En una reunión": "in a business meeting pose",
    "Trabajando en laptop": "working on a laptop",
    "De picnic": "having a picnic, sitting on a blanket",
    "Cuidando un caballo": "grooming a horse",
}

# ── Pose dinámica / de acción (distinta de la pose de cuerpo estática) ──
POSE_DINAMICA = {
    "Sin especificar": "",
    "Saltando (en el aire)": "jumping in the air, dynamic pose",
    "Salto lateral": "jumping sideways, dynamic pose",
    "Salto alto con brazos arriba": "jumping high with arms raised, dynamic pose",
    "Corriendo de frente": "running towards viewer, dynamic pose",
    "Corriendo de lado": "running, side view, dynamic pose",
    "Corriendo de espaldas": "running away, back view, dynamic pose",
    "Sprint (máxima velocidad)": "sprinting at full speed, motion lines",
    "Bailando (vista frontal)": "dancing pose, front view",
    "Bailando (vista lateral)": "dancing pose, side view",
    "Bailando con energía": "energetic dance pose",
    "Celebrando": "celebrating pose, fists up",
    "Con actitud": "confident attitude pose, hand on hip",
    "Lista para la acción": "action-ready crouching pose",
    "Selfie con V de la paz": "taking a selfie, holding phone, peace sign",
    "Selfie divertida": "taking a fun selfie, holding phone",
    "Selfie desde arriba": "taking a selfie from a high angle, holding phone",
    "En el aire (vista abajo)": "jumping in the air, low angle view from below",
    "Puños al frente": "fists forward, boxing stance, dynamic pose",
    "Rugiendo": "roaring battle cry pose",
    "Energía (aura)": "glowing power aura effect around character, dynamic energy pose",
    "Superhéroe": "heroic superhero landing pose",
    "Salto con patineta": "skateboard jump trick pose, with skateboard",
    "Parkour": "parkour vaulting pose",
    "Saltando la cuerda": "jumping rope pose, with jump rope",
    "Andando en bicicleta": "riding a bicycle pose, with bicycle",
    "Con fuerza": "flexing muscles, showing strength pose",
    "Hablando": "talking, mid-speech hand gesture",
    "Agachado": "crouching pose",
}

# ── Actividad / vehículo (funciona mejor con 'Imagen con fondo', ya que
# trae su propio contexto de escena) ──
ACTIVIDAD_VEHICULO = {
    "Sin especificar": "",
    "En moto (frente)": "riding a motorcycle, front view, wearing a motorcycle helmet",
    "En moto (lado)": "riding a motorcycle, side view",
    "En moto (espalda)": "riding a motorcycle, back view",
    "En moto (sonriendo, sin casco)": "sitting on a parked motorcycle, smiling, no helmet, peace sign",
    "En moto (en acción)": "riding a motorcycle at speed, dynamic action shot, motion lines",
    "En patineta (de pie)": "standing on a skateboard",
    "En patineta (en movimiento)": "riding a skateboard in motion, hair flowing",
    "En patineta (salto/trick)": "skateboard jump trick, dynamic pose",
    "En bicicleta (frente)": "riding a bicycle, front view, wearing a helmet",
    "En bicicleta (espalda)": "riding a bicycle, back view",
    "En bicicleta (subida, esfuerzo)": "riding a bicycle uphill, effort expression",
    "En bicicleta (casco y gafas)": "riding a bicycle, helmet and sunglasses",
    "En auto (conduciendo)": "driving a car, hands on the steering wheel",
    "En auto (mirando por la ventana)": "looking out of a car window",
    "En auto (de pasajera/o)": "sitting as a passenger in a car",
    "En avión (ventanilla)": "sitting by an airplane window, headphones on",
    "En avión (uniforme de piloto)": "wearing pilot uniform, airport tarmac background",
    "En helicóptero (cabina)": "sitting in a helicopter cabin, headset on",
    "En helicóptero (piloto)": "piloting a helicopter, headset on, thumbs up",
    "Montando a caballo": "horseback riding",
}

# ── Escenarios preset para el modo 'Imagen con fondo' ──
ESCENARIO_PRESET = {
    "Personalizado (uso el texto de abajo)": None,
    "Colegio": "school hallway background",
    "Ciudad": "city street background",
    "Parque": "park background, trees, daylight",
    "Casa": "cozy home interior background",
    "Gimnasio": "gym interior background",
    "Playa (de día, ropa normal)": "beach background, daylight, boardwalk",
    "Piscina": "swimming pool background, daylight",
    "Ciudad de noche": "city street at night, neon lights background",
    "Cocina": "kitchen interior background",
    "Navidad": "christmas tree and lights background",
    "Nieve/invierno": "snowy winter background",
    "Río/naturaleza": "river nature background",
    "Bosque": "forest background, trees, dappled sunlight",
    "Atardecer": "sunset background, warm lighting",
    "Oficina": "office interior background",
    "Consultorio médico": "medical clinic background",
    "Obra de construcción": "construction site background",
    "Aula/salón de clases": "classroom background",
    "Establo": "horse stable background",
    "Pradera/campo abierto": "open countryside meadow background",
    "Jardín": "garden background",
}

# ── Tipo de salida: sticker (fondo transparente) vs imagen normal (con fondo) ──
TIPO_SALIDA = {
    "Sticker (fondo transparente)": "sticker",
    "Imagen con fondo": "con_fondo",
}
# Para el modo sticker generamos siempre sobre fondo BLANCO liso (más fácil y
# más limpio de recortar con rembg) y recién después lo convertimos a
# transparente en post-proceso — no le pedimos "transparent background" al
# modelo de imagen porque no sabe generar canal alfa, solo simularía un fondo
# ajedrezado o gris que después ensucia el recorte.
ESTILO_ARTE_BASE = "manga anime style, clean lineart, cel shading, high detail"
ESTILO_SIN_FONDO = f"{ESTILO_ARTE_BASE}, sticker style, plain solid white background, no scenery"
FONDO_POR_DEFECTO_CON_ESCENA = "simple anime background, soft depth of field"

NEGATIVO_UNIVERSAL = (
    "lowres, blurry, extra fingers, deformed hands, bad anatomy, watermark, text, "
    "signature, nsfw, nudity, sexualized minor, child in swimwear, suggestive pose on minor"
)
NEGATIVO_EXTRA_SIN_FONDO = "gradient background, patterned background, scenery, shadow on background"

# ──────────────────────────────────────────────────────────────────────────
# Guardrail para "Generación libre" (Módulo/pestaña 3): ahí el texto es
# 100% libre, sin los dropdowns que en el resto de la app ya evitan por
# diseño ciertas combinaciones (como el freno de ROPA_SOLO_ADULTO). Esto es
# una red adicional, no un reemplazo del prompt negativo — bloquea de plano
# la generación si el prompt combina términos de menor de edad con términos
# sexuales/de desnudez, en vez de solo pedirle al modelo que lo evite.
# ──────────────────────────────────────────────────────────────────────────
_TERMINOS_MENOR = {
    "child", "kid", "toddler", "baby", "infant", "minor", "underage", "preteen",
    "niño", "niña", "nino", "nina", "infante", "bebé", "bebe", "menor de edad",
}
_TERMINOS_SEXUALES = {
    "nude", "naked", "nsfw", "sex", "sexual", "porn", "erotic", "provocative",
    "lingerie", "desnud", "sexo", "erótic", "erotic", "provocativ",
}


def _prompt_libre_es_seguro(texto):
    t = texto.lower()
    tiene_menor = any(term in t for term in _TERMINOS_MENOR)
    tiene_sexual = any(term in t for term in _TERMINOS_SEXUALES)
    return not (tiene_menor and tiene_sexual)


def construir_prompt(token_identificador, descripcion_base, genero, edad,
                      color_ojos, tipo_ojos, color_pelo, largo_pelo, estilo_pelo,
                      accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
                      mascota, collar_mascota, color_collar, actividad_mascota,
                      expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
                      tipo_salida, escenario_preset, fondo_descripcion):
    if tipo_ropa in ROPA_SOLO_ADULTO and edad != "Adulto":
        # Freno duro: esto no depende de qué tan bien el modelo respete el
        # prompt negativo. Si la combinación no es válida, no se genera.
        raise gr.Error(
            f"'{tipo_ropa}' solo está disponible con Edad = Adulto. "
            f"Elegí otra prenda para esta edad."
        )

    partes = [token_identificador, descripcion_base, GENERO[genero], EDAD[edad],
              COLOR_OJOS[color_ojos], TIPO_OJOS[tipo_ojos],
              COLOR_PELO[color_pelo], LARGO_PELO[largo_pelo],
              f"{ESTILO_PELO[estilo_pelo]} hair"]

    accesorios_frag = ", ".join(ACCESORIOS[a] for a in accesorios_lista if ACCESORIOS[a])
    if accesorios_frag:
        partes.append(accesorios_frag)

    ropa_frag = TIPO_ROPA[tipo_ropa]
    if color_ropa and color_ropa.strip():
        ropa_frag = f"{color_ropa.strip()} {ropa_frag}"
    partes.append(ropa_frag)
    partes.append(CALZADO[calzado])

    if PROFESION[profesion]:
        partes.append(PROFESION[profesion])

    if MASCOTA[mascota]:
        frag_mascota = MASCOTA[mascota]
        if COLLAR_MASCOTA[collar_mascota]:
            collar_frag = COLLAR_MASCOTA[collar_mascota]
            if color_collar and color_collar.strip():
                collar_frag = collar_frag.replace("collar", f"{color_collar.strip()} collar", 1)
            frag_mascota = f"{frag_mascota}, {collar_frag}"
        partes.append(frag_mascota)
        if ACTIVIDAD_MASCOTA[actividad_mascota]:
            partes.append(ACTIVIDAD_MASCOTA[actividad_mascota])

    if EXPRESION[expresion]:
        partes.append(EXPRESION[expresion])
    if GESTO[gesto]:
        partes.append(GESTO[gesto])
    if VISTA[vista]:
        partes.append(VISTA[vista])
    if POSE_CUERPO[pose_cuerpo]:
        partes.append(POSE_CUERPO[pose_cuerpo])
    if POSE_DINAMICA[pose_dinamica]:
        partes.append(POSE_DINAMICA[pose_dinamica])
    if ACTIVIDAD_VEHICULO[actividad_vehiculo]:
        partes.append(ACTIVIDAD_VEHICULO[actividad_vehiculo])
    if pose_texto and pose_texto.strip():
        partes.append(pose_texto.strip())

    if TIPO_SALIDA[tipo_salida] == "sticker":
        partes.append(ESTILO_SIN_FONDO)
    else:
        partes.append(ESTILO_ARTE_BASE)
        preset = ESCENARIO_PRESET.get(escenario_preset)
        if preset:
            partes.append(preset)
        elif fondo_descripcion and fondo_descripcion.strip():
            partes.append(fondo_descripcion.strip())
        else:
            partes.append(FONDO_POR_DEFECTO_CON_ESCENA)

    return ", ".join(p for p in partes if p)


# ──────────────────────────────────────────────────────────────────────────
# Quitar el fondo (solo para el modo sticker) — rembg, corre en CPU, liviano.
# ──────────────────────────────────────────────────────────────────────────
_sesion_rembg = None


def _cargar_rembg():
    global _sesion_rembg
    if _sesion_rembg is None:
        from rembg import new_session
        _sesion_rembg = new_session("isnet-anime")  # modelo de rembg afinado para arte anime/ilustración
    return _sesion_rembg


def quitar_fondo(imagen_pil):
    from rembg import remove
    sesion = _cargar_rembg()
    return remove(imagen_pil, session=sesion)  # devuelve RGBA con canal alfa real


def agregar_borde_sticker(imagen_rgba, grosor_frac=0.014):
    """Contorno blanco tipo 'die-cut' alrededor de la silueta ya recortada
    (el look clásico de sticker de mensajería, como en tu referencia):
    dilata la máscara alfa, la rellena de blanco sólido, y pega la imagen
    original encima. grosor_frac es proporcional al lado más chico de la
    imagen para que el grosor del borde se vea consistente sin importar la
    resolución de salida."""
    arr = np.array(imagen_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    grosor_px = max(4, int(min(arr.shape[0], arr.shape[1]) * grosor_frac))
    kernel = np.ones((grosor_px, grosor_px), np.uint8)
    alpha_dilatado = cv2.dilate(alpha, kernel, iterations=1)

    capa_borde = np.zeros_like(arr)
    capa_borde[:, :, 0:3] = 255  # blanco sólido
    capa_borde[:, :, 3] = alpha_dilatado

    fondo_borde = Image.fromarray(capa_borde, mode="RGBA")
    return Image.alpha_composite(fondo_borde, imagen_rgba.convert("RGBA"))


# ──────────────────────────────────────────────────────────────────────────
# Control de calidad — detección de manos de más (mediapipe)
#
# HONESTO SOBRE LAS LIMITACIONES: este modelo está entrenado con fotos reales
# de manos, no con arte anime — puede fallar en cualquier dirección (no
# detectar manos que sí están bien, o detectar formas raras del dibujo como
# si fueran una mano). Por eso el chequeo es ASIMÉTRICO a propósito: solo
# trata como sospechoso detectar CLARAMENTE MÁS de 2 manos (señal fuerte de
# duplicación/deformidad) — nunca falla por detectar 0 o 1 (son poses
# normales: manos ocultas, de espaldas, etc.). No es perfecto ni reemplaza
# mirar la imagen vos, pero atrapa el caso que preguntaste.
# ──────────────────────────────────────────────────────────────────────────
_detector_manos = None
_RUTA_MODELO_MANOS = os.path.join(CARPETA_SALIDAS, "hand_landmarker.task")
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
        print(f"⚠️ No se pudo cargar el detector de manos (control de calidad desactivado): {e}", flush=True)
        _detector_manos = False  # marca que falló, para no reintentar cada vez
    return _detector_manos


def contar_manos(imagen_pil):
    """Devuelve la cantidad de manos detectadas, o None si el detector no
    está disponible (sin internet, falló la descarga, etc.)."""
    import mediapipe as mp
    detector = _cargar_detector_manos()
    if detector is False:
        return None
    arr = np.array(imagen_pil.convert("RGB"))
    mp_imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    resultado = detector.detect(mp_imagen)
    return len(resultado.hand_landmarks)


# ──────────────────────────────────────────────────────────────────────────
# Control de calidad — cabezas de más (cascada Haar/LBP específica para
# caras anime, no la genérica de fotos reales — mucho más confiable acá).
# Misma filosofía asimétrica: solo desconfiar de MÁS DE 1 cara detectada.
# ──────────────────────────────────────────────────────────────────────────
_cascada_caras_anime = None
_RUTA_CASCADA_ANIME = os.path.join(CARPETA_SALIDAS, "lbpcascade_animeface.xml")
_URL_CASCADA_ANIME = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"


def _cargar_cascada_caras():
    global _cascada_caras_anime
    if _cascada_caras_anime is not None:
        return _cascada_caras_anime
    try:
        if not os.path.exists(_RUTA_CASCADA_ANIME):
            subprocess.run(["wget", "-q", "-O", _RUTA_CASCADA_ANIME, _URL_CASCADA_ANIME], check=True)
        cascada = cv2.CascadeClassifier(_RUTA_CASCADA_ANIME)
        if cascada.empty():
            raise RuntimeError("La cascada se descargó pero no cargó bien (archivo corrupto)")
        _cascada_caras_anime = cascada
    except Exception as e:
        print(f"⚠️ No se pudo cargar el detector de caras anime: {e}", flush=True)
        _cascada_caras_anime = False
    return _cascada_caras_anime


def contar_cabezas(imagen_pil):
    """Devuelve la cantidad de caras anime detectadas, o None si el
    detector no está disponible. Parámetros conservadores a propósito
    (probamos varias combinaciones): priorizan NO marcar falsos positivos
    por duplicar la misma cara, a costa de a veces no detectar ninguna."""
    cascada = _cargar_cascada_caras()
    if cascada is False:
        return None
    arr = np.array(imagen_pil.convert("RGB"))
    gris = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gris = cv2.equalizeHist(gris)
    lado_min = int(min(arr.shape[0], arr.shape[1]) * 0.05)
    caras = cascada.detectMultiScale(gris, scaleFactor=1.05, minNeighbors=4, minSize=(lado_min, lado_min))
    return len(caras)


# ──────────────────────────────────────────────────────────────────────────
# Control de calidad — piernas/pies de más, por silueta (no hay un
# "detector de piernas" open source confiable para esto). Se apoya en el
# recorte de fondo (rembg) que ya usamos para el modo sticker: mira una
# franja horizontal cerca del piso en la máscara alfa y cuenta cuántos
# bloques separados de silueta hay ahí — cada bloque debería ser un pie o
# una pierna. Misma filosofía asimétrica: solo desconfiar de MÁS DE 2.
# ──────────────────────────────────────────────────────────────────────────
def contar_bloques_piernas(imagen_rgba):
    arr = np.array(imagen_rgba.convert("RGBA"))
    alpha = arr[:, :, 3]
    h, _ = alpha.shape
    y0, y1 = int(h * 0.90), int(h * 0.96)  # franja cerca de los pies
    franja = (alpha[y0:y1] > 128).astype(np.uint8)
    if franja.size == 0:
        return None
    columna = franja.max(axis=0)
    # Cierre morfológico horizontal: un hueco de 1-4px suele ser ruido de
    # rembg (anti-aliasing/sombra), no una pierna separada de verdad — sin
    # esto, ese ruido parte una sola pierna en dos "bloques" (falso positivo
    # que de hecho encontramos probando esto).
    columna_2d = (columna * 255).reshape(1, -1).astype(np.uint8)
    columna_cerrada = cv2.morphologyEx(columna_2d, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    columna_final = (columna_cerrada[0] > 0).astype(int)
    transiciones = np.diff(columna_final)
    bloques = int((transiciones == 1).sum()) + (1 if columna_final[0] else 0)
    return bloques


# ──────────────────────────────────────────────────────────────────────────
# Chequeo combinado — usado tanto por el generador como por el agente que
# organiza el dataset (más abajo / en agente_calidad_dataset.py).
# ──────────────────────────────────────────────────────────────────────────
UMBRAL_MANOS = 2
UMBRAL_CABEZAS = 1
UMBRAL_PIERNAS = 2


def revisar_calidad_imagen(imagen_pil, imagen_rgba_sin_fondo=None):
    """Corre los 3 chequeos. imagen_rgba_sin_fondo es opcional (para poder
    reutilizar un recorte ya hecho); si no se pasa, no se chequean piernas
    (necesita el canal alfa).
    Devuelve (ok: bool, motivos: list[str], detector_faltante: bool)."""
    motivos = []
    detector_faltante = False

    manos = contar_manos(imagen_pil)
    if manos is None:
        detector_faltante = True
    elif manos > UMBRAL_MANOS:
        motivos.append(f"{manos} manos detectadas")

    cabezas = contar_cabezas(imagen_pil)
    if cabezas is None:
        detector_faltante = True
    elif cabezas > UMBRAL_CABEZAS:
        motivos.append(f"{cabezas} caras detectadas")

    if imagen_rgba_sin_fondo is not None:
        piernas = contar_bloques_piernas(imagen_rgba_sin_fondo)
        if piernas is not None and piernas > UMBRAL_PIERNAS:
            motivos.append(f"{piernas} bloques de pierna/pie detectados")

    return len(motivos) == 0, motivos, detector_faltante


# ──────────────────────────────────────────────────────────────────────────
# PASO 5 — Generar la imagen/sticker con los atributos elegidos
# ──────────────────────────────────────────────────────────────────────────
def generar_personaje(nombre_personaje, token_identificador, descripcion_base,
                       genero, edad, color_ojos, tipo_ojos, color_pelo, largo_pelo, estilo_pelo,
                       accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
                       mascota, collar_mascota, color_collar, actividad_mascota,
                       expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
                       tipo_salida, escenario_preset, fondo_descripcion, agregar_borde,
                       activar_control_calidad, semilla, num_imagenes, progress=gr.Progress()):
    progress(0.1, desc="Cargando personaje...")
    pipe = _cargar_pipeline(nombre_personaje)

    prompt = construir_prompt(token_identificador, descripcion_base, genero, edad,
                               color_ojos, tipo_ojos, color_pelo, largo_pelo, estilo_pelo,
                               accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
                               mascota, collar_mascota, color_collar, actividad_mascota,
                               expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
                               tipo_salida, escenario_preset, fondo_descripcion)
    es_sticker = TIPO_SALIDA[tipo_salida] == "sticker"
    negativo = NEGATIVO_UNIVERSAL + (f", {NEGATIVO_EXTRA_SIN_FONDO}" if es_sticker else "")
    print(f"[Generación] Prompt: {prompt}", flush=True)

    semilla_base = int(semilla) if semilla not in (None, "", -1) else random.randint(0, 2**31 - 1)
    MAX_INTENTOS_POR_IMAGEN = 3

    rutas = []
    avisos = []
    algun_detector_no_disponible = False

    for i in range(int(num_imagenes)):
        semilla_intento = semilla_base + i
        img_final = None
        rgba_final = None  # el recorte sin fondo, si ya lo calculamos para el chequeo de piernas
        for intento in range(1, MAX_INTENTOS_POR_IMAGEN + 1):
            progress(0.15 + 0.7 * (i / max(1, int(num_imagenes))),
                     desc=f"Generando imagen {i+1}/{int(num_imagenes)} (intento {intento})...")
            generador = torch.Generator(device="cpu").manual_seed(semilla_intento)
            resultado = pipe(prompt=prompt, negative_prompt=negativo, num_inference_steps=30,
                              num_images_per_prompt=1, generator=generador, height=1024, width=1024)
            candidato = resultado.images[0]

            if activar_control_calidad:
                # El recorte de fondo se necesita para el chequeo de piernas — lo calculamos
                # una vez acá y lo reusamos como salida final si el modo es sticker, para no
                # correr rembg dos veces sobre la misma imagen.
                rgba_candidato = quitar_fondo(candidato)
                ok, motivos, detector_faltante = revisar_calidad_imagen(candidato, rgba_candidato)
                if detector_faltante:
                    algun_detector_no_disponible = True
            else:
                ok, motivos, rgba_candidato = True, [], None

            if ok:
                img_final, rgba_final = candidato, rgba_candidato
                if intento > 1:
                    avisos.append(f"Imagen {i+1}: se corrigió sola en el intento {intento} "
                                   f"(semilla {semilla_intento}).")
                break
            print(f"⚠️ Imagen {i+1}, intento {intento}: {', '.join(motivos)} (semilla {semilla_intento}). "
                  f"Reintentando con otra semilla.", flush=True)
            semilla_intento = semilla_base + 1000 * intento + i  # semilla bien distinta para el reintento
        else:
            avisos.append(f"⚠️ Imagen {i+1}: después de {MAX_INTENTOS_POR_IMAGEN} intentos sigue marcando "
                           f"{', '.join(motivos)} (semilla final {semilla_intento}) — revisala a ojo.")

        if img_final is None:  # se agotaron los intentos, usar el último candidato igual
            img_final, rgba_final = candidato, rgba_candidato

        if es_sticker:
            img_final = rgba_final if rgba_final is not None else quitar_fondo(img_final)
            if agregar_borde:
                img_final = agregar_borde_sticker(img_final)
        ruta = os.path.join(CARPETA_SALIDAS, f"{nombre_personaje}_{semilla_intento}_{i+1}.png")
        img_final.save(ruta)
        rutas.append(ruta)

    if algun_detector_no_disponible:
        avisos.insert(0, "ℹ️ Algún detector de control de calidad no pudo cargarse (¿sin internet en esta "
                          "sesión?). Puede haber corrido con chequeos parciales o ninguno.")

    log = "✅ Listo, sin anomalías detectadas." if not avisos else "\n".join(avisos)
    progress(1.0, desc="¡Listo!")
    return rutas, semilla_base, log


# ──────────────────────────────────────────────────────────────────────────
# PASO 6 — Generación libre (sin personaje/LoRA): el modelo base de
# Qwen-Image con lo que el usuario escriba, para escenas que no son "el
# personaje de marca" — otra persona, un animal, cualquier cosa.
# ──────────────────────────────────────────────────────────────────────────
FORMATOS_LIBRE = {
    "Cuadrado (1024x1024)": (1024, 1024),
    "Vertical (1024x1536)": (1024, 1536),
    "Horizontal (1536x1024)": (1536, 1024),
}


def generar_libre(prompt, prompt_negativo_extra, formato, quitar_fondo_bool,
                   semilla, num_imagenes, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Escribí una descripción de lo que querés generar.")
    if not _prompt_libre_es_seguro(prompt):
        # No damos detalle de qué combinación de palabras disparó esto —
        # ver la nota sobre no explicar mecánica de detección de seguridad.
        raise gr.Error("Ese pedido no se puede generar.")

    progress(0.1, desc="Cargando modelo base (sin personaje)...")
    pipe = _cargar_pipeline_sin_lora()

    ancho, alto = FORMATOS_LIBRE[formato]
    negativo = NEGATIVO_UNIVERSAL
    if prompt_negativo_extra and prompt_negativo_extra.strip():
        negativo = f"{negativo}, {prompt_negativo_extra.strip()}"

    semilla_usada = int(semilla) if semilla not in (None, "", -1) else random.randint(0, 2**31 - 1)
    generador = torch.Generator(device="cpu").manual_seed(semilla_usada)

    progress(0.4, desc=f"Generando {int(num_imagenes)} imagen(es)...")
    resultado = pipe(
        prompt=prompt.strip(), negative_prompt=negativo,
        num_inference_steps=30, num_images_per_prompt=int(num_imagenes),
        generator=generador, height=alto, width=ancho,
    )

    rutas = []
    for i, img in enumerate(resultado.images):
        if quitar_fondo_bool:
            img = quitar_fondo(img)
        ruta = os.path.join(CARPETA_SALIDAS, f"libre_{semilla_usada}_{i+1}.png")
        img.save(ruta)
        rutas.append(ruta)

    progress(1.0, desc=f"Listo. Semilla usada: {semilla_usada}.")
    return rutas, semilla_usada


# ──────────────────────────────────────────────────────────────────────────
# Interfaz Gradio — tres pestañas: Entrenar personaje / Generar imágenes /
# Generación libre
# ──────────────────────────────────────────────────────────────────────────
def _wrapper_agregar_dataset(nombre, archivos, descripcion_base, es_grilla, progress=gr.Progress()):
    return agregar_al_dataset(nombre, archivos, descripcion_base, es_grilla, progress=progress)


def _wrapper_entrenar(nombre, descripcion_base, pasos, rank, progress=gr.Progress()):
    nombre = nombre.strip().lower().replace(" ", "_")
    if not nombre:
        raise gr.Error("Ponele un nombre corto al personaje (ej: 'ren', 'kai').")
    carpeta = os.path.join(CARPETA_DATASETS, nombre)
    n_imagenes = len([f for f in os.listdir(carpeta) if f.endswith(".png")]) if os.path.isdir(carpeta) else 0
    if n_imagenes < 3:
        raise gr.Error(
            f"Solo hay {n_imagenes} imagen(es) en el dataset de '{nombre}'. "
            f"Agregá al menos 3-5 con el botón de arriba antes de entrenar."
        )
    token = f"{nombre}_chr"
    entrenar_lora_personaje(nombre, carpeta, token, pasos=int(pasos), rank=int(rank), progress=progress)
    return (f"✅ LoRA de '{nombre}' entrenado con {n_imagenes} imágenes. Token identificador: {token}\n"
            f"Andá a la pestaña 'Generar' y usá ese mismo nombre y token.")


def _wrapper_generar(nombre, token, descripcion_base, genero, edad, color_ojos, tipo_ojos, color_pelo,
                      largo_pelo, estilo_pelo, accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
                      mascota, collar_mascota, color_collar, actividad_mascota,
                      expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
                      tipo_salida, escenario_preset, fondo_descripcion, agregar_borde, activar_control_calidad,
                      semilla, num_imagenes, progress=gr.Progress()):
    nombre = nombre.strip().lower().replace(" ", "_")
    rutas, semilla_usada, log = generar_personaje(
        nombre, token, descripcion_base, genero, edad, color_ojos, tipo_ojos, color_pelo,
        largo_pelo, estilo_pelo, accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
        mascota, collar_mascota, color_collar, actividad_mascota,
        expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
        tipo_salida, escenario_preset, fondo_descripcion, agregar_borde, activar_control_calidad,
        semilla, num_imagenes, progress=progress,
    )
    return rutas, semilla_usada, log


def _alternar_campo_fondo(tipo_salida):
    """Muestra los controles de fondo (preset + texto libre) solo cuando la
    salida elegida es 'con fondo'; el checkbox de borde sticker es al revés,
    solo tiene sentido en modo sticker."""
    es_con_fondo = TIPO_SALIDA[tipo_salida] == "con_fondo"
    return gr.update(visible=es_con_fondo), gr.update(visible=es_con_fondo), gr.update(visible=not es_con_fondo)


with gr.Blocks(title="Generador de Personajes Manga/Anime") as interfaz:
    gr.Markdown("# 🎨 Generador de Personajes — Manga/Anime (Qwen-Image + LoRA)\n"
                "Módulo 1 del pipeline. Entrená un personaje una vez, después generá "
                "todas las variantes de atributos que quieras.")

    with gr.Tab("1️⃣ Entrenar personaje"):
        nombre_e = gr.Textbox(label="Nombre corto del personaje (sin espacios/acentos)", placeholder="ren")
        descripcion_e = gr.Textbox(
            label="Descripción base fija del personaje (rasgos que NO cambian entre generaciones)",
            placeholder="ej: sharp jawline, small mole under left eye, slim build",
        )

        gr.Markdown("### 📥 Agregar imágenes al dataset\n"
                    "Podés hacer esto varias veces, en cualquier momento — cada tanda se suma a las "
                    "anteriores, no las reemplaza.")
        archivos_e = gr.File(label="Subí imágenes del personaje", file_count="multiple", type="filepath")
        es_grilla_e = gr.Checkbox(
            value=False,
            label="Es una hoja/grilla con varias poses en una sola imagen (la recorto automáticamente en paneles)")
        boton_agregar = gr.Button("📥 Agregar al dataset")
        log_agregar = gr.Textbox(label="Resultado", interactive=False, lines=3)
        boton_agregar.click(_wrapper_agregar_dataset, [nombre_e, archivos_e, descripcion_e, es_grilla_e], log_agregar)

        gr.Markdown("### 🎓 Entrenar\nUsa todo lo que ya esté acumulado en el dataset de este personaje "
                    "(subido ahora o en sesiones anteriores).")
        with gr.Row():
            pasos_e = gr.Slider(400, 3000, value=1200, step=100, label="Pasos de entrenamiento")
            rank_e = gr.Slider(8, 64, value=32, step=8, label="Rank del LoRA (más alto = más fidelidad, más VRAM)")
        boton_entrenar = gr.Button("🎓 Entrenar LoRA", variant="primary")
        log_entrenamiento = gr.Textbox(label="Estado", interactive=False)
        boton_entrenar.click(_wrapper_entrenar, [nombre_e, descripcion_e, pasos_e, rank_e], log_entrenamiento)

    with gr.Tab("2️⃣ Generar imágenes"):
        with gr.Row():
            nombre_g = gr.Textbox(label="Nombre del personaje (igual al usado al entrenar)", placeholder="ren")
            token_g = gr.Textbox(label="Token identificador", placeholder="ren_chr")
        descripcion_g = gr.Textbox(label="Descripción base fija (misma que usaste al entrenar)")

        with gr.Row():
            genero_g = gr.Dropdown(list(GENERO), value="Mujer", label="Género")
            edad_g = gr.Dropdown(list(EDAD), value="Adulto", label="Edad")

        with gr.Row():
            ojos_g = gr.Dropdown(list(COLOR_OJOS), value="Marrón", label="Color de ojos")
            tipo_ojos_g = gr.Dropdown(list(TIPO_OJOS), value="Grandes clásicos", label="Tipo/forma de ojos")
            pelo_color_g = gr.Dropdown(list(COLOR_PELO), value="Negro", label="Color de pelo")
            pelo_largo_g = gr.Dropdown(list(LARGO_PELO), value="Mediano", label="Largo de pelo")
            pelo_estilo_g = gr.Dropdown(list(ESTILO_PELO), value="Liso", label="Estilo de pelo")

        accesorios_g = gr.CheckboxGroup(list(ACCESORIOS), value=["Ninguno"], label="Accesorios")

        with gr.Row():
            ropa_g = gr.Dropdown(list(TIPO_ROPA), value="Casual",
                                  label="Tipo de ropa (los trajes de baño solo funcionan con Edad = Adulto)")
            color_ropa_g = gr.Textbox(label="Color de ropa (opcional, texto libre)", placeholder="ej: rojo y negro")
            calzado_g = gr.Dropdown(list(CALZADO), value="Zapatillas/tenis", label="Calzado")
            profesion_g = gr.Dropdown(list(PROFESION), value="Ninguna", label="Profesión/rol (opcional)")

        with gr.Row():
            mascota_g = gr.Dropdown(list(MASCOTA), value="Ninguna", label="Mascota (opcional)")
            collar_mascota_g = gr.Dropdown(list(COLLAR_MASCOTA), value="Ninguno", label="Collar de la mascota")
            color_collar_g = gr.Textbox(label="Color del collar (opcional)", placeholder="ej: negro")
            actividad_mascota_g = gr.Dropdown(list(ACTIVIDAD_MASCOTA), value="Sin especificar",
                                               label="Interacción con la mascota")

        with gr.Row():
            expresion_g = gr.Dropdown(list(EXPRESION), value="Ninguna (uso el texto de pose libre)",
                                       label="Expresión facial")
            gesto_g = gr.Dropdown(list(GESTO), value="Ninguno", label="Gesto de manos")
            vista_g = gr.Dropdown(list(VISTA), value="Sin especificar",
                                   label="Ángulo/vista (para turnaround)")

        with gr.Row():
            pose_cuerpo_g = gr.Dropdown(list(POSE_CUERPO), value="Sin especificar",
                                         label="Pose de cuerpo (estática)")
            pose_dinamica_g = gr.Dropdown(list(POSE_DINAMICA), value="Sin especificar",
                                           label="Pose dinámica (salto/corrida/baile/selfie)")
            actividad_vehiculo_g = gr.Dropdown(list(ACTIVIDAD_VEHICULO), value="Sin especificar",
                                                label="Actividad/vehículo (mejor con 'Imagen con fondo')")

        pose_g = gr.Textbox(label="Pose libre / detalle extra (se suma a lo de arriba)",
                             placeholder="ej: apoyada en una pared, con mochila")

        with gr.Row():
            tipo_salida_g = gr.Radio(list(TIPO_SALIDA), value="Sticker (fondo transparente)",
                                      label="Tipo de salida")
            borde_sticker_g = gr.Checkbox(value=True, label="Agregar contorno blanco tipo sticker (die-cut)")
            control_calidad_g = gr.Checkbox(
                value=True,
                label="Control de calidad: reintentar si detecta manos/cabezas/piernas de más (heurístico, más lento)")
            escenario_preset_g = gr.Dropdown(list(ESCENARIO_PRESET), value="Personalizado (uso el texto de abajo)",
                                              label="Escenario preset", visible=False)
            fondo_desc_g = gr.Textbox(
                label="Descripción del fondo (si elegiste 'Personalizado' arriba)",
                placeholder="ej: salón de clases, atardecer en la ciudad, bosque",
                visible=False,
            )
        tipo_salida_g.change(_alternar_campo_fondo, tipo_salida_g, [escenario_preset_g, fondo_desc_g, borde_sticker_g])

        with gr.Row():
            semilla_g = gr.Number(label="Semilla (vacío = aleatoria; usá la misma semilla para variar un solo atributo)", value=None)
            num_imagenes_g = gr.Slider(1, 4, value=1, step=1, label="Cantidad de imágenes")

        boton_generar = gr.Button("Generar", variant="primary")
        galeria = gr.Gallery(label="Resultado", columns=2)
        semilla_usada_g = gr.Number(label="Semilla base usada (guardala)", interactive=False)
        log_calidad_g = gr.Textbox(label="Control de calidad", interactive=False)

        boton_generar.click(
            _wrapper_generar,
            [nombre_g, token_g, descripcion_g, genero_g, edad_g, ojos_g, tipo_ojos_g, pelo_color_g,
             pelo_largo_g, pelo_estilo_g, accesorios_g, ropa_g, calzado_g, color_ropa_g, profesion_g,
             mascota_g, collar_mascota_g, color_collar_g, actividad_mascota_g,
             expresion_g, gesto_g, vista_g, pose_cuerpo_g, pose_dinamica_g, actividad_vehiculo_g, pose_g,
             tipo_salida_g, escenario_preset_g, fondo_desc_g, borde_sticker_g, control_calidad_g,
             semilla_g, num_imagenes_g],
            [galeria, semilla_usada_g, log_calidad_g],
        )

    with gr.Tab("3️⃣ Generación libre"):
        gr.Markdown(
            "Sin personaje ni LoRA — el modelo base de Qwen-Image con lo que escribas. "
            "Para cualquier escena que no sea el personaje de marca: otra persona, un "
            "animal, un objeto, lo que sea. **Se aplican las mismas restricciones de "
            "seguridad que en el resto de la app** (nada de contenido sexual, y menos "
            "aún involucrando menores — eso no se puede desactivar acá)."
        )
        prompt_libre_g = gr.Textbox(
            label="Descripción (en inglés funciona mejor, pero se puede en español)",
            placeholder="ej: a black dog with a collar, a woman with blue eyes and short "
                        "reddish hair walking next to it, park background, anime style",
            lines=3,
        )
        negativo_libre_g = gr.Textbox(
            label="Qué evitar (opcional, se suma a las restricciones de seguridad fijas)",
            placeholder="ej: blurry, extra limbs",
        )
        with gr.Row():
            formato_libre_g = gr.Dropdown(list(FORMATOS_LIBRE), value="Cuadrado (1024x1024)", label="Formato")
            quitar_fondo_libre_g = gr.Checkbox(value=False, label="Quitar el fondo (recorte automático)")

        with gr.Row():
            semilla_libre_g = gr.Number(label="Semilla (vacío = aleatoria)", value=None)
            num_imagenes_libre_g = gr.Slider(1, 4, value=1, step=1, label="Cantidad de imágenes")

        boton_generar_libre = gr.Button("Generar", variant="primary")
        galeria_libre = gr.Gallery(label="Resultado", columns=2)
        semilla_usada_libre_g = gr.Number(label="Semilla usada (guardala)", interactive=False)

        boton_generar_libre.click(
            generar_libre,
            [prompt_libre_g, negativo_libre_g, formato_libre_g, quitar_fondo_libre_g,
             semilla_libre_g, num_imagenes_libre_g],
            [galeria_libre, semilla_usada_libre_g],
        )

if __name__ == "__main__":
    interfaz.launch(share=True)
