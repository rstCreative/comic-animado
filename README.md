# Cómic Animado — Interfaz simple

Interfaz web (Gradio) para convertir una página de cómic/webcómic en un video
narrado en 4K, con voces naturales por personaje. 100% open source.

## Qué hace el botón "Generar Video"
1. Detecta y recorta los paneles automáticamente (sobre la imagen original, sin alterar).
2. Sube la resolución y aviva los colores de cada panel (Real-ESRGAN, sin tocar el contenido).
3. Lee los diálogos (OCR) y reconoce quién habla en cada panel (reconocimiento facial).
4. Genera una voz natural distinta por personaje (XTTS-v2).
5. **Anima cada panel de verdad**: primeros planos con boca sincronizada al audio (SadTalker), y escenas con movimiento de fondo/pelo/luces (AnimateDiff). Si alguno de los dos falla en un panel puntual (son los modelos más pesados/frágiles del pipeline), ese panel cae automáticamente al modo de respaldo (Ken Burns: zoom/paneo sobre la imagen fija) — así un fallo puntual no arruina el video completo.
6. Escala el resultado final a 4K real.

> La primera vez que uses la interfaz, los pasos 5 y 6 descargan varios modelos grandes (SadTalker, AnimateDiff, Real-ESRGAN) — puede tardar bastante. Las siguientes veces es más rápido porque ya quedan en caché.

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
- La primera ejecución tarda más: descarga el modelo de voces (XTTS-v2), SadTalker,
  AnimateDiff y Real-ESRGAN. Las siguientes veces es más rápido.
- Si `face_recognition` falla al instalar, necesitas `cmake` en el sistema
  primero: `apt-get install -y cmake`.
- **Sobre la capa generativa (SadTalker/AnimateDiff):** son los modelos más
  pesados y menos maduros del pipeline. Si fallan en algún panel, ese panel
  usa automáticamente el modo de respaldo (Ken Burns) — revisa los mensajes
  en la consola de Colab (empiezan con ⚠️) para ver si esto ocurrió.
- Recomendado GPU con 16GB+ VRAM para correr todo (voces + generativa + 4K)
  sin quedarte sin memoria.
