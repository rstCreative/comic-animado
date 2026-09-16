"""
Módulo de corrección interactiva para Qwen-Image + LoRA.

Flujo:
1) Generar una imagen.
2) Revisarla y escribir la anomalía observada (ej. "dos cabezas").
3) Regenerar conservando personaje/escena y reforzando la corrección.
4) Guardar la corrección en memoria persistente (JSONL).
5) Las correcciones acumuladas pueden usarse después como material de
   revisión/reentrenamiento; una corrección aislada NO modifica el LoRA.

Se ejecuta sobre las funciones/pipeline existentes de app.py y guarda la
memoria en Google Drive cuando Drive está montado.
"""

import json
import os
import random
import traceback
from datetime import datetime, timezone

import gradio as gr
import torch
from PIL import Image

import app


BASE_MEMORIA = os.path.join(
    app._BASE_DRIVE, "memoria_generacion"
) if app._DRIVE_MONTADO else "memoria_generacion"

os.makedirs(BASE_MEMORIA, exist_ok=True)

ANOMALIAS = {
    "Dos cabezas / cabeza duplicada": "two heads, exactly one head, one face, anatomically correct head",
    "Manos de más / manos deformes": "exactly two hands when visible, correct hand anatomy, no extra hands, no fused fingers",
    "Piernas de más / piernas deformes": "exactly two legs when visible, correct leg anatomy, no extra legs",
    "Brazos de más / brazos deformes": "exactly two arms, correct arm anatomy, no extra arms",
    "Dedos de más": "correct number of fingers, natural finger anatomy, no extra fingers",
    "Ojos/cara deformados": "symmetrical anime face, two eyes, natural facial anatomy",
    "Cuerpo deformado": "correct human anatomy, natural proportions, coherent limbs and joints",
    "Parte del cuerpo fusionada": "separated coherent limbs and body parts, clean anatomy",
    "Otro": "correct anatomy, coherent body structure",
}


def _ruta_memoria(personaje):
    personaje = (personaje or "sin_personaje").strip().lower().replace(" ", "_")
    carpeta = os.path.join(BASE_MEMORIA, personaje)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def _guardar_memoria(personaje, registro):
    ruta = os.path.join(_ruta_memoria(personaje), "errores.jsonl")
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return ruta


def _normalizar_entrada(entrada):
    nombre, descripcion = app._parsear_entrada_personaje(entrada)
    token = f"{nombre}_chr"
    prompt = ", ".join([
        token,
        descripcion,
        app.ESTILO_ARTE_BASE,
    ])
    return nombre, descripcion, prompt


def _generar(prompt, personaje, semilla, salida="con_fondo"):
    pipe = app._cargar_pipeline(personaje)
    semilla = int(semilla) if semilla not in (None, "", -1) else random.randint(0, 2**31 - 1)
    generador = torch.Generator(device="cpu").manual_seed(semilla)
    negativo = app.NEGATIVO_UNIVERSAL + ", extra limbs, duplicated body parts, malformed anatomy"
    resultado = pipe(
        prompt=prompt,
        negative_prompt=negativo,
        num_inference_steps=30,
        num_images_per_prompt=1,
        generator=generador,
        height=1024,
        width=1024,
    )
    img = resultado.images[0]
    ruta = os.path.join(app.CARPETA_SALIDAS, f"feedback_{personaje}_{semilla}.png")
    img.save(ruta)
    return img, ruta, semilla


def _revisar(img):
    try:
        rgba = app.quitar_fondo(img)
        ok, motivos, detector_faltante = app.revisar_calidad_imagen(img, rgba)
        if detector_faltante:
            return ok, motivos, "⚠️ Algún detector no está disponible; la revisión es parcial."
        return ok, motivos, ""
    except Exception as e:
        return True, [], f"ℹ️ No se pudo ejecutar todo el control automático: {e}"


def generar_inicial(entrada, semilla, progress=gr.Progress()):
    try:
        personaje, descripcion, prompt = _normalizar_entrada(entrada)
        progress(0.2, desc="Cargando LoRA y generando...")
        img, ruta, semilla_usada = _generar(prompt, personaje, semilla)
        ok, motivos, nota = _revisar(img)
        estado = "✅ Generación inicial lista."
        if motivos:
            estado += " ⚠️ " + "; ".join(motivos)
        if nota:
            estado += "\n" + nota
        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tipo": "generacion",
            "personaje": personaje,
            "prompt": prompt,
            "semilla": semilla_usada,
            "ruta_original": ruta,
            "anomalías_detectadas": motivos,
        }
        _guardar_memoria(personaje, registro)
        progress(1.0, desc="Listo")
        return img, personaje, prompt, semilla_usada, estado
    except gr.Error:
        raise
    except Exception:
        return None, None, None, None, "❌ Error:\n\n" + traceback.format_exc()


def corregir_y_regenerar(entrada, personaje, prompt_original, semilla_original,
                         tipo_anomalia, descripcion_error, conservar_escena,
                         conservar_ropa, conservar_personaje, progress=gr.Progress()):
    if not personaje or not prompt_original:
        raise gr.Error("Primero generá una imagen.")
    if not descripcion_error or not descripcion_error.strip():
        raise gr.Error("Escribí qué anomalía querés corregir.")

    correccion = ANOMALIAS.get(tipo_anomalia, ANOMALIAS["Otro"])
    texto_usuario = descripcion_error.strip()
    bloques = [
        "CORRECTION PASS",
        prompt_original,
        f"Fix this specific problem: {texto_usuario}.",
        correccion,
        "single coherent character, anatomically correct, clean anime illustration",
    ]
    if conservar_personaje:
        bloques.append("preserve the same character identity, face, hair, accessories and visual design")
    if conservar_ropa:
        bloques.append("preserve the same clothing and colors")
    if conservar_escena:
        bloques.append("preserve the same scene, camera framing and composition")

    prompt_corregido = ", ".join(bloques)
    semilla_nueva = (int(semilla_original) + 7919) if semilla_original not in (None, "") else None

    progress(0.2, desc="Aplicando corrección y regenerando...")
    img, ruta, semilla_usada = _generar(prompt_corregido, personaje, semilla_nueva)
    ok, motivos, nota = _revisar(img)

    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tipo": "correccion",
        "personaje": personaje,
        "prompt_original": prompt_original,
        "prompt_corregido": prompt_corregido,
        "semilla_original": semilla_original,
        "semilla_nueva": semilla_usada,
        "tipo_anomalia": tipo_anomalia,
        "descripcion_error": texto_usuario,
        "conservar_personaje": bool(conservar_personaje),
        "conservar_ropa": bool(conservar_ropa),
        "conservar_escena": bool(conservar_escena),
        "ruta_corregida": ruta,
        "anomalías_despues": motivos,
    }
    ruta_memoria = _guardar_memoria(personaje, registro)

    estado = "✅ Regenerada con la corrección indicada."
    if motivos:
        estado += "\n⚠️ El control automático todavía detecta: " + "; ".join(motivos)
    if nota:
        estado += "\n" + nota
    estado += f"\n🧠 Corrección guardada en memoria: {ruta_memoria}"
    progress(1.0, desc="Corrección terminada")
    return img, prompt_corregido, semilla_usada, estado


with gr.Blocks(title="Corrección inteligente — Qwen-Image + LoRA") as interfaz_feedback:
    gr.Markdown(
        "# 🔧 Corrección inteligente\n"
        "Generá → indicá la anomalía → regenerá conservando lo que estaba bien. "
        "Las correcciones se guardan para detectar errores recurrentes."
    )

    entrada = gr.Textbox(
        label="Personaje y escena",
        placeholder="rstchica: caminando en un bosque, cuerpo completo",
        lines=3,
    )
    semilla = gr.Number(label="Semilla (vacío = aleatoria)", value=None)
    boton_generar = gr.Button("🎨 Generar", variant="primary")
    resultado = gr.Image(label="Resultado", type="pil")

    personaje_state = gr.State(value=None)
    prompt_state = gr.State(value=None)
    semilla_state = gr.State(value=None)
    estado = gr.Textbox(label="Estado", lines=5, interactive=False)

    gr.Markdown("## Si hay un error")
    tipo_anomalia = gr.Dropdown(list(ANOMALIAS), value="Otro", label="Tipo de anomalía")
    descripcion_error = gr.Textbox(
        label="¿Qué querés corregir?",
        placeholder="Ejemplo: generó dos cabezas; debe tener una sola cabeza y mantener la misma cara.",
        lines=3,
    )
    with gr.Row():
        conservar_personaje = gr.Checkbox(value=True, label="Conservar personaje")
        conservar_ropa = gr.Checkbox(value=True, label="Conservar ropa")
        conservar_escena = gr.Checkbox(value=True, label="Conservar escena/composición")
    boton_corregir = gr.Button("🔁 CORREGIR Y REGENERAR", variant="primary")

    boton_generar.click(
        generar_inicial,
        [entrada, semilla],
        [resultado, personaje_state, prompt_state, semilla_state, estado],
    )
    boton_corregir.click(
        corregir_y_regenerar,
        [entrada, personaje_state, prompt_state, semilla_state, tipo_anomalia,
         descripcion_error, conservar_escena, conservar_ropa, conservar_personaje],
        [resultado, prompt_state, semilla_state, estado],
    )


if __name__ == "__main__":
    interfaz_feedback.launch(share=True)
