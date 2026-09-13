# Cómic Animado — Interfaz simple

Interfaz web (Gradio) para convertir una página de cómic/webcómic en un video
narrado en 4K, con voces naturales por personaje. 100% open source.

## Qué hace el botón "Generar Video"
1. Sube tu resolución y aviva los colores (sin alterar el dibujo).
2. Detecta y recorta los paneles automáticamente.
3. Lee los diálogos (OCR) y reconoce quién habla en cada panel (reconocimiento facial).
4. Genera una voz natural distinta por personaje (XTTS-v2).
5. Anima cada panel (Ken Burns) y lo escala a 4K real (Real-ESRGAN).

## Requisitos
- Python 3.10+
- GPU recomendada (no obligatoria, pero mucho más lento en CPU). El escalado a
  4K usa PyTorch/CUDA directo — funciona en cualquier plataforma con GPU NVIDIA,
  no depende de ejecutables específicos de sistema operativo.
- `ffmpeg` instalado en el sistema (`apt-get install ffmpeg` en Linux).

## Uso en Google Colab
```python
!pip install -q -r requirements.txt
!apt-get install -y -qq ffmpeg
!python app.py
```
Aparecerá un link (`*.gradio.live`) — ábrelo en el navegador.

## Uso fuera de Colab (tu PC, RunPod, Vast.ai, etc.)
```bash
sudo apt-get install -y ffmpeg   # una sola vez
pip install -r requirements.txt
python app.py
```
Se abre automáticamente un link local (`http://127.0.0.1:7860`) y uno público
temporal (`*.gradio.live`) que puedes compartir o abrir desde el celular.

## Notas
- La primera ejecución tarda más: descarga el modelo de voces (XTTS-v2) y el
  ejecutable de escalado (Real-ESRGAN). Las siguientes veces es más rápido.
- Si `face_recognition` falla al instalar, necesitas `cmake` en el sistema
  primero: `apt-get install -y cmake`.
- Esta interfaz cubre el pipeline base (paneles + diálogos + voces + animación
  + 4K). La capa generativa avanzada (SadTalker/AnimateDiff) no está incluida
  aquí a propósito, para que el botón sea siempre confiable.
