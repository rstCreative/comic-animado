# Generador de Personajes Manga/Anime (Qwen-Image + LoRA)

Interfaz web (Gradio) para entrenar un LoRA de personaje sobre **Qwen-Image**
y generar imágenes de ese personaje describiendo todo por texto. 100% open
source. Pensado para correr en **Google Colab**.

> Este repo reemplaza la versión anterior del proyecto (que partía de una
> página de cómic ya existente y la animaba). El enfoque cambió a generar
> personajes consistentes desde cero con un LoRA propio.

## Qué incluye

| Archivo | Qué hace |
|---|---|
| `app.py` | La interfaz Gradio: 3 pestañas (entrenar, generar con personaje, generación libre) |
| `requirements.txt` | Dependencias para instalar en Colab |
| `preparar_dataset_grillas.py` | Utilidad de línea de comandos independiente: recorta una hoja/grilla de referencia en paneles y clasifica por OCR entre varios personajes a la vez (para tandas grandes fuera de la interfaz) |
| `agente_calidad_dataset.py` | Utilidad de línea de comandos independiente: re-revisa un dataset entero ya armado y mueve lo sospechoso a `revisar_manual/` |

## Uso en Colab

```python
!apt-get install -y -qq tesseract-ocr tesseract-ocr-spa  # binario que necesita pytesseract
!pip install -q --upgrade torchao                          # evita choque de versión con diffusers
!pip install -q -r requirements.txt
!python app.py
```

Aparecerá un link (`*.gradio.live`) — ábrelo en el navegador.

### 1️⃣ Entrenar personaje

Todo automático: escribís el nombre del personaje y subís imágenes con el
botón **➕**. Por cada subida:

1. Si la imagen es una hoja/grilla con varias poses, se recorta sola en
   paneles individuales (no hace falta aclararlo — si es una foto suelta,
   se usa tal cual).
2. Cada imagen resultante pasa el control de calidad (manos/cabezas/piernas
   de más); lo sospechoso va a `personajes_dataset/revisar_manual/<nombre>/`
   en vez de al dataset de entrenamiento.
3. Si ya hay 3 o más imágenes limpias acumuladas para ese personaje, se
   **reentrena el LoRA automáticamente**, sin ningún botón de "Entrenar"
   aparte.

⚠️ Como se reentrena en cada subida (a partir de la 3ª imagen), subir de a
una implica un entrenamiento completo cada vez. Para entrenar una sola vez,
subí todas las imágenes juntas en una misma tanda.

### 2️⃣ Generar imágenes

Una sola casilla, formato `nombre_personaje: descripción libre`:

```
rstchica: sentada en la playa comiendo un helado, con un perro de collar negro al lado
```

Todo lo de antes de los `:` es el nombre (el mismo que usaste al entrenar);
todo lo de después es una descripción libre de lo que querés generar — ropa,
pose, fondo, mascota, lo que sea, en una sola frase. El resto de las
opciones (sticker vs. imagen con fondo, semilla, control de calidad) están
en el acordeón "⚙️ Opciones de salida", plegado por default.

### 3️⃣ Generación libre

Sin personaje ni LoRA — el modelo base de Qwen-Image con lo que escribas.
Para cualquier escena que no sea el personaje entrenado.

## Requisitos de hardware

- **Generar** (inferencia): anda con el T4 gratis de Colab.
- **Entrenar** un LoRA: Qwen-Image es un modelo grande — probablemente
  necesites Colab Pro con GPU A100 (40GB), o cuantización de 4-bit. Ver los
  comentarios en `app.py` (`entrenar_lora_personaje`) para las banderas de
  ahorro de memoria (`--use_8bit_adam`, `--offload`, `cache_latents`).

## Si algo falla

Los errores ya no se muestran como una caja vacía de "Error": aparecen como
texto completo (traceback de Python) en la caja de estado de cada pestaña.
Copiá ese texto tal cual si necesitás ayuda para diagnosticarlo.

## Límites del contenido que genera

El generador no incluye —bajo ninguna combinación ni escribiéndolo en el
prompt— ropa interior/lencería, prendas tipo tanga, ni poses o expresiones
sexuales o provocativas. Hay un filtro de palabras clave que bloquea
directamente cualquier pedido que combine términos de menor de edad con
términos sexuales o de ropa reveladora, tanto en el modo de atributos como
en el modo de prompt libre — no es una sugerencia al modelo, corta la
generación antes de que empiece.
