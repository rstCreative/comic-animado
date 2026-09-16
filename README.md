# Generador de Personajes Manga/Anime (Qwen-Image + LoRA)

Interfaz web (Gradio) para entrenar un LoRA de personaje sobre **Qwen-Image**
y generar stickers/imágenes de ese personaje controlando atributos por
dropdown: apariencia, ropa, expresión, pose, escenario. 100% open source.
Pensado para correr en **Google Colab**.

> Este repo reemplaza la versión anterior del proyecto (que partía de una
> página de cómic ya existente y la animaba). El enfoque cambió a generar
> personajes consistentes desde cero con un LoRA propio.

## Qué incluye

| Archivo | Qué hace |
|---|---|
| `app.py` | La interfaz Gradio: pestaña de entrenamiento del LoRA + pestaña de generación |
| `requirements.txt` | Dependencias para instalar en Colab |
| `preparar_dataset_grillas.py` | Utilidad: recorta automáticamente paneles individuales de una hoja/grilla de referencia (turnarounds, poses, expresiones) y arma las carpetas de dataset con captions listas para entrenar |
| `agente_calidad_dataset.py` | Utilidad: revisa un dataset ya armado en busca de manos/cabezas/piernas de más, y mueve lo sospechoso a `revisar_manual/` sin tocar el resto |

## Uso en Colab

```python
!apt-get install -y -qq tesseract-ocr tesseract-ocr-spa  # binario que necesita pytesseract
!pip install -q -r requirements.txt
!python app.py
```

Aparecerá un link (`*.gradio.live`) — ábrelo en el navegador.

### 1. Entrenar un personaje

En la pestaña **"1️⃣ Entrenar personaje"**: subí 5-15 imágenes del personaje
en distintas posturas (esto es lo que le enseña al LoRA a generar poses
nuevas más adelante), ponele un nombre corto, y entrená.

Si tenés una hoja de referencia tipo grilla (turnarounds, expresiones,
poses) en vez de imágenes sueltas, corré primero:

```bash
python preparar_dataset_grillas.py
```

(ajustá las rutas de entrada dentro del script a tus archivos) para
recortarla en imágenes individuales con captions automáticos antes de
entrenar.

Antes de lanzar el entrenamiento, conviene correr:

```bash
python agente_calidad_dataset.py personajes_dataset
```

para sacar del dataset cualquier imagen con manos/cabezas/piernas de más
antes de que el LoRA la aprenda.

### 2. Generar imágenes

En la pestaña **"2️⃣ Generar imágenes"**: elegís atributos (ojos, pelo,
ropa, expresión, pose, escenario) y generás. El control de calidad
automático reintenta con otra semilla si detecta una anomalía obvia.

## Requisitos de hardware

- **Generar** (inferencia): anda con el T4 gratis de Colab.
- **Entrenar** un LoRA: Qwen-Image es un modelo grande — probablemente
  necesites Colab Pro con GPU A100 (40GB), o cuantización de 4-bit. Ver los
  comentarios en `app.py` (`entrenar_lora_personaje`) para las banderas de
  ahorro de memoria (`--use_8bit_adam`, `--offload`, `cache_latents`).

## Límites del contenido que genera

El generador no incluye —bajo ninguna combinación de atributos— ropa
interior/lencería, prendas tipo tanga, ni poses o expresiones sexuales o
provocativas. Las prendas de vestido de baño están bloqueadas por código
(no solo por prompt) si la Edad seleccionada no es "Adulto".
