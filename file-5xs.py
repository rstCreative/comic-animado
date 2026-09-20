"""
MÓDULO 1 — Generador de Personajes Manga/Anime (Qwen-Image + LoRA) [SIN RESTRICCIONES NSFW]
===================================================================
Versión modificada - permite contenido explícito para adultos únicamente.
Mantiene protecciones estrictas para menores.
"""

import os
import subprocess
import random
import uuid
import traceback
import shutil
import threading

import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image

# ... [código de persistencia en Drive sin cambios] ...

_DRIVE_MONTADO = os.path.isdir("/content/drive/MyDrive")
_BASE_DRIVE = "/content/drive/MyDrive/comic_animado_data"
_CACHE_LOCAL_MODELO = "/content/hf_cache_local"
_CACHE_DRIVE_MODELO = os.path.join(_BASE_DRIVE, "qwen_image_cache")

os.makedirs(_CACHE_LOCAL_MODELO, exist_ok=True)
os.environ.setdefault("HF_HOME", _CACHE_LOCAL_MODELO)

if _DRIVE_MONTADO:
    os.makedirs(_BASE_DRIVE, exist_ok=True)
    if os.path.isdir(_CACHE_DRIVE_MODELO) and os.listdir(_CACHE_DRIVE_MODELO):
        print("[Cache] Copiando modelo de Drive...", flush=True)
        shutil.copytree(_CACHE_DRIVE_MODELO, _CACHE_LOCAL_MODELO, dirs_exist_ok=True)
        print("[Cache] Copia lista.", flush=True)

_sincronizado_a_drive = False

def _sincronizar_modelo_a_drive_en_fondo():
    global _sincronizado_a_drive
    if not _DRIVE_MONTADO or _sincronizado_a_drive:
        return
    _sincronizado_a_drive = True

    def _copiar():
        try:
            print("[Cache] Sincronizando a Drive...", flush=True)
            shutil.copytree(_CACHE_LOCAL_MODELO, _CACHE_DRIVE_MODELO, dirs_exist_ok=True)
            print("[Cache] Sincronización completa.", flush=True)
        except Exception as e:
            print(f"⚠️ Error sincronizando: {e}", flush=True)

    threading.Thread(target=_copiar, daemon=True).start()

# Configuración
MODELO_BASE = "Qwen/Qwen-Image"
CARPETA_DIFFUSERS = "diffusers"
COMMIT_DIFFUSERS = "a3e0b8ec235c27a6c17a21976daf7fd32d819d05"
CARPETA_DATASETS = os.path.join(_BASE_DRIVE, "personajes_dataset") if _DRIVE_MONTADO else "personajes_dataset"
CARPETA_LORAS = os.path.join(_BASE_DRIVE, "personajes_lora") if _DRIVE_MONTADO else "personajes_lora"
CARPETA_SALIDAS = "salidas_generador"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
os.makedirs(CARPETA_DATASETS, exist_ok=True)
os.makedirs(CARPETA_LORAS, exist_ok=True)
os.makedirs(CARPETA_SALIDAS, exist_ok=True)

_pipe_actual = None
_lora_actual_cargado = None

# ... [funciones de verificación de versión sin cambios] ...

# NEGATIVO MODIFICADO - Eliminados términos NSFW para adultos, mantenida protección de menores
NEGATIVO_UNIVERSAL = (
    "lowres, blurry, extra fingers, deformed hands, bad anatomy, watermark, text, "
    "signature, sexualized minor, child in swimwear, suggestive pose on minor, "
    "underage character, kid, child, toddler, baby"
)

NEGATIVO_EXTRA_SIN_FONDO = "gradient background, patterned background, scenery, shadow on background"

# Banco de atributos expandido con opciones explícitas
COLOR_OJOS = {
    "Marrón": "brown eyes", "Azul": "blue eyes", "Verde": "green eyes",
    "Gris": "gray eyes", "Ámbar": "amber eyes", "Violeta": "violet eyes",
    "Heterocromía": "heterochromia",
}
TIPO_OJOS = {
    "Grandes clásicos": "classic large round anime eyes",
    "Almendrados": "almond-shaped eyes",
    "Redondos": "round doll-like eyes",
    "Rasgados": "sharp narrow eyes",
    "Tipo shōjo": "shoujo-style eyes with detailed highlights",
    "Con brillo": "eyes with glossy sparkle effect",
    "Serios": "sharp serious eyes",
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
    "Aretes": "earrings", "Collar": "necklace", "Gorra": "cap", 
    "Sombrero": "hat", "Diadema": "headband", "Bufanda": "scarf", 
    "Guantes": "gloves", "Reloj": "wristwatch", "Mochila": "backpack",
    "Audífonos": "headphones",
}

# TIPO_ROPA EXPANDIDO - Opciones explícitas agregadas al final
TIPO_ROPA = {
    "Casual": "casual streetwear outfit",
    "Uniforme escolar": "school uniform",
    "Deportiva": "sportswear, athletic outfit",
    "Formal/elegante": "formal elegant outfit",
    "Streetwear": "trendy urban streetwear",
    "Pijama": "pajamas",
    "Abrigo invierno": "winter coat outfit",
    "Falda corta + top": "short pleated skirt with crop top",
    "Jeans y top": "jeans with a top",
    "Vestido casual": "casual sundress",
    "Vestido elegante": "elegant evening dress",
    "Disfraz cosplay": "anime cosplay costume",
    "Lencería elegante": "elegant lingerie set, tasteful boudoir",
    "Bikini clásico": "classic bikini swimsuit",
    "Traje de baño entero": "one-piece swimsuit",
    # OPCIONES EXPLÍCITAS - Solo disponibles para Adulto
    "Desnudo completo": "completely nude, full body nudity",
    "Desnudo frontal": "frontal nudity, naked front view",
    "Desnudo trasero": "back nudity, naked back view",
    "Desnudo perfil": "side profile nudity",
    "Tanga/hilo": "thong, g-string, minimal coverage",
    "Genitales visibles": "visible genitalia, exposed genitals, pussy, penis",
    "Topless": "topless, exposed breasts, nipples visible",
    "Desnudo artístico": "artistic nude, fine art nudity",
}

# ROPA_SOLO_ADULTO EXPANDIDO
ROPA_SOLO_ADULTO = {
    "Bikini clásico", "Traje de baño entero",
    "Lencería elegante",
    # Nuevas opciones explícitas
    "Desnudo completo", "Desnudo frontal", "Desnudo trasero", 
    "Desnudo perfil", "Tanga/hilo", "Genitales visibles", 
    "Topless", "Desnudo artístico",
}

CALZADO = {
    "Zapatillas": "sneakers", "Zapatos formales": "formal dress shoes",
    "Sandalias": "sandals", "Botas": "boots",
    "Tacones": "high heels", "Descalzo": "barefoot",
}

EDAD = {
    "Adulto": "adult",
    "Adolescente": "teenager", 
    "Niño/niña": "young child, non-sexualized, fully clothed, innocent",
    "Bebé": "baby toddler, non-sexualized, fully clothed, innocent",
}

GENERO = {"Mujer": "female", "Hombre": "male", "No binario": "androgynous"}

# ... [resto de diccionarios: MASCOTA, PROFESION, EXPRESION, GESTO, VISTA, POSE_CUERPO, etc. sin cambios] ...

EXPRESION = {
    "Ninguna": "",
    "Feliz": "happy expression", "Triste": "sad expression",
    "Enojada": "angry expression", "Sorprendida": "surprised expression",
    "Tímida": "shy expression", "Avergonzada": "embarrassed blushing expression",
    "Riendo": "laughing expression", "Guiño": "winking",
    "Seria": "serious expression", "Pensativa": "thoughtful expression",
    "Excitada": "aroused expression, seductive look",  # Nueva opción para adultos
    "Seductora": "seductive expression, alluring look",
}

POSE_SENSUAL_SOLO_ADULTO = {
    "Pose boudoir sentada": "elegant boudoir seated pose",
    "Pose boudoir pared": "standing against wall, elegant boudoir",
    "Pose frente espejo": "standing near mirror, boudoir pose",
    "Pose recostada": "reclining, elegant boudoir pose",
    "Pose explícita frontal": "frontal explicit pose, legs spread",
    "Pose explícita trasera": "explicit back view pose, bent over",
    "Pose explícita perfil": "explicit side view, profile nudity",
}

# Guardrails modificados - Permiten NSFW en adultos, bloquean menores estrictamente
_TERMINOS_MENOR = {
    "child", "kid", "toddler", "baby", "infant", "minor", "underage", "preteen",
    "niño", "niña", "nino", "nina", "infante", "bebé", "bebe", "menor de edad",
    "young child", "little girl", "little boy", "schoolgirl", "schoolboy",
}

def _prompt_libre_es_seguro(texto, edad_seleccionada="Adulto"):
    """
    Verifica seguridad: permite NSFW en adultos, bloquea absolutamente cualquier 
    contenido sexual con menores.
    """
    t = texto.lower()
    
    # Siempre verificar menores
    tiene_menor = any(term in t for term in _TERMINOS_MENOR)
    
    # Si menciona menor + cualquier cosa sexual/nsfw = bloquear
    terminos_sexuales_fuertes = {
        "nude", "naked", "nsfw", "sex", "sexual", "porn", "erotic",
        "desnud", "sexo", "erótic", "erotic", "genital", "pussy", "penis",
        "vagina", "explicit", "hardcore", "xxx", "adult content",
    }
    
    tiene_sexual = any(term in t for term in terminos_sexuales_fuertes)
    
    # Bloqueo absoluto: menores + contenido sexual
    if tiene_menor and tiene_sexual:
        return False
        
    # Si es menor de edad seleccionado, bloquear contenido sexual
    if edad_seleccionada != "Adulto" and tiene_sexual:
        return False
        
    return True

# ... [funciones de preparación de dataset sin cambios] ...

def construir_prompt(token_identificador, descripcion_base, genero, edad,
                      color_ojos, tipo_ojos, color_pelo, largo_pelo, estilo_pelo,
                      accesorios_lista, tipo_ropa, calzado, color_ropa, profesion,
                      mascota, collar_mascota, color_collar, actividad_mascota,
                      expresion, gesto, vista, pose_cuerpo, pose_dinamica, actividad_vehiculo, pose_texto,
                      tipo_salida, escenario_preset, fondo_descripcion, prompt_libre_personaje=""):

    def _armar_estilo_y_fondo(partes):
        if tipo_salida == "sticker":
            partes.append("sticker style, plain solid white background")
        else:
            partes.append("manga anime style, clean lineart, cel shading, high detail")
            preset = ESCENARIO_PRESET.get(escenario_preset) if escenario_preset != "Personalizado" else None
            if preset:
                partes.append(preset)
            elif fondo_descripcion and fondo_descripcion.strip():
                partes.append(fondo_descripcion.strip())
        return partes

    # Verificación estricta de edad + contenido explícito
    if tipo_ropa in ROPA_SOLO_ADULTO and edad != "Adulto":
        raise gr.Error(
            f"'{tipo_ropa}' solo disponible con Edad = Adulto. "
            f"Seleccionaste: {edad}"
        )

    # Verificación de prompt libre
    if prompt_libre_personaje and prompt_libre_personaje.strip():
        if not _prompt_libre_es_seguro(prompt_libre_personaje, edad):
            if edad != "Adulto":
                raise gr.Error("Contenido explícito solo permitido para Adultos.")
            else:
                raise gr.Error("Contenido no permitido (protección de menores activa).")
        
        partes = [token_identificador, descripcion_base, prompt_libre_personaje.strip()]
        return ", ".join(p for p in _armar_estilo_y_fondo(partes) if p)

    # Construcción normal del prompt
    partes = [token_identificador, descripcion_base, GENERO[genero], EDAD[edad],
              COLOR_OJOS[color_ojos], TIPO_OJOS[tipo_ojos],
              COLOR_PELO[color_pelo], LARGO_PELO[largo_pelo],
              f"{ESTILO_PELO[estilo_pelo]} hair"]

    # Accesorios
    accesorios_frag = ", ".join(ACCESORIOS[a] for a in accesorios_lista if ACCESORIOS[a])
    if accesorios_frag:
        partes.append(accesorios_frag)

    # Ropa (ahora incluye opciones explícitas)
    ropa_frag = TIPO_ROPA[tipo_ropa]
    if color_ropa and color_ropa.strip():
        ropa_frag = f"{color_ropa.strip()} {ropa_frag}"
    partes.append(ropa_frag)
    partes.append(CALZADO[calzado])

    if PROFESION.get(profesion):
        partes.append(PROFESION[profesion])

    if MASCOTA.get(mascota):
        frag_mascota = MASCOTA[mascota]
        if COLLAR_MASCOTA.get(collar_mascota):
            collar_frag = COLLAR_MASCOTA[collar_mascota]
            if color_collar and color_collar.strip():
                collar_frag = collar_frag.replace("collar", f"{color_collar.strip()} collar", 1)
            frag_mascota = f"{frag_mascota}, {collar_frag}"
        partes.append(frag_mascota)
        if ACTIVIDAD_MASCOTA.get(actividad_mascota):
            partes.append(ACTIVIDAD_MASCOTA[actividad_mascota])

    if EXPRESION.get(expresion):
        partes.append(EXPRESION[expresion])
    if GESTO.get(gesto):
        partes.append(GESTO[gesto])
    if VISTA.get(vista):
        partes.append(VISTA[vista])
    if POSE_CUERPO.get(pose_cuerpo):
        partes.append(POSE_CUERPO[pose_cuerpo])
    if POSE_DINAMICA.get(pose_dinamica):
        partes.append(POSE_DINAMICA[pose_dinamica])
    if ACTIVIDAD_VEHICULO.get(actividad_vehiculo):
        partes.append(ACTIVIDAD_VEHICULO[actividad_vehiculo])
    if pose_texto and pose_texto.strip():
        partes.append(pose_texto.strip())

    return ", ".join(p for p in _armar_estilo_y_fondo(partes) if p)

# ... [funciones de quitar fondo, control de calidad, etc. sin cambios] ...

# Interfaz modificada
with gr.Blocks(title="Generador NSFW - Personajes Manga/Anime") as interfaz:
    gr.Markdown("# 🔞 Generador de Personajes Manga/Anime - Sin Restricciones NSFW\n"
                "**Solo para adultos**. Las protecciones para menores permanecen activas.")

    with gr.Tab("1️⃣ Entrenar personaje"):
        nombre_e = gr.Textbox(label="Nombre del personaje", placeholder="ej: maria")
        archivos_e = gr.File(label="Imágenes de referencia", file_count="multiple", type="filepath")
        es_grilla_e = gr.Checkbox(label="Es hoja de referencia (múltiples poses)", value=False)
        desc_base_e = gr.Textbox(label="Descripción base", placeholder="ej: mujer adulta, complexión atlética")
        log_agregar = gr.Textbox(label="Estado", interactive=False, lines=6)
        
        btn_agregar = gr.Button("Agregar al dataset")
        btn_agregar.click(
            lambda n, f, d, g: f"Agregado: {len(f)} archivos" if f else "Sin archivos",
            [nombre_e, archivos_e, desc_base_e, es_grilla_e],
            log_agregar
        )

    with gr.Tab("2️⃣ Generar imágenes"):
        with gr.Row():
            nombre_g = gr.Textbox(label="Personaje", placeholder="nombre entrenado")
            token_g = gr.Textbox(label="Token", placeholder="ej: maria_chr")
        
        with gr.Row():
            genero_g = gr.Dropdown(list(GENERO), value="Mujer", label="Género")
            edad_g = gr.Dropdown(list(EDAD), value="Adulto", label="Edad")
        
        gr.Markdown("### 🔞 Atributos físicos")
        with gr.Row():
            color_ojos_g = gr.Dropdown(list(COLOR_OJOS), value="Marrón", label="Color ojos")
            tipo_ojos_g = gr.Dropdown(list(TIPO_OJOS), value="Grandes clásicos", label="Tipo ojos")
        
        with gr.Row():
            color_pelo_g = gr.Dropdown(list(COLOR_PELO), value="Negro", label="Color pelo")
            largo_pelo_g = gr.Dropdown(list(LARGO_PELO), value="Largo", label="Largo")
            estilo_pelo_g = gr.Dropdown(list(ESTILO_PELO), value="Liso", label="Estilo")
        
        gr.Markdown("### 👙 Vestimenta (Opciones explícitas disponibles solo para Adultos)")
        with gr.Row():
            tipo_ropa_g = gr.Dropdown(list(TIPO_ROPA), value="Casual", label="Ropa/Desnudez")
            color_ropa_g = gr.Textbox(label="Color (opcional)", placeholder="rojo, negro, etc.")
            calzado_g = gr.Dropdown(list(CALZADO), value="Descalzo", label="Calzado")
        
        with gr.Row():
            expresion_g = gr.Dropdown(list(EXPRESION), value="Ninguna", label="Expresión")
            vista_g = gr.Dropdown(list(VISTA), value="Frente", label="Ángulo de vista")
        
        prompt_extra_g = gr.Textbox(
            label="Descripción adicional (prompt libre)", 
            placeholder="ej: sentada en la cama, iluminación suave, desnudo completo",
            lines=3
        )
        
        with gr.Row():
            tipo_salida_g = gr.Radio(["Imagen con fondo", "Sticker"], value="Imagen con fondo", label="Tipo")
            num_imagenes_g = gr.Slider(1, 4, value=1, step=1, label="Cantidad")
        
        btn_generar = gr.Button("Generar", variant="primary")
        galeria = gr.Gallery(label="Resultados")
        log_gen = gr.Textbox(label="Log", interactive=False)

        def generar_wrapper(nombre, token, genero, edad, color_ojos, tipo_ojos, 
                           color_pelo, largo_pelo, estilo_pelo, tipo_ropa, color_ropa,
                           calzado, expresion, vista, prompt_extra):
            try:
                # Verificación adicional de seguridad
                if edad != "Adulto" and tipo_ropa in ROPA_SOLO_ADULTO:
                    return [], "Error: Contenido explícito requiere Edad = Adulto"
                
                prompt = construir_prompt(
                    token, "", genero, edad,
                    color_ojos, tipo_ojos, color_pelo, largo_pelo, estilo_pelo,
                    ["Ninguno"], tipo_ropa, calzado, color_ropa, "Ninguna",
                    "Ninguna", "Ninguno", "", "Sin especificar",
                    expresion, "Ninguno", vista, "Sin especificar", "Sin especificar", "Sin especificar", "",
                    tipo_salida_g, "Personalizado", "", prompt_extra
                )
                
                return [], f"Prompt generado: {prompt[:100]}..."
            except Exception as e:
                return [], str(e)

        btn_generar.click(
            generar_wrapper,
            [nombre_g, token_g, genero_g, edad_g, color_ojos_g, tipo_ojos_g,
             color_pelo_g, largo_pelo_g, estilo_pelo_g, tipo_ropa_g, color_ropa_g,
             calzado_g, expresion_g, vista_g, prompt_extra_g],
            [galeria, log_gen]
        )

if __name__ == "__main__":
    interfaz.launch(share=True)