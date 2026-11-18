# ClipForge Studio

Aplicación Streamlit para convertir videos en clips verticales 9:16 con subtítulos automáticos.

## Funciones
- Entrada mediante enlace de YouTube o archivo de video.
- Transcripción con Faster-Whisper.
- Selección de momentos con Gemini.
- Exportación vertical con subtítulos y descarga ZIP.

## Requisitos del sistema
- Python 3.10 o superior.
- FFmpeg y ffprobe instalados y disponibles en PATH.
- Una clave API válida de Google Gemini.

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

La clave de Gemini se introduce en el panel lateral de la aplicación. No la publiques en el repositorio ni la compartas.

## Despliegue
La aplicación requiere FFmpeg/ffprobe y recursos de CPU y memoria para procesar video y ejecutar Whisper. Se recomienda un servicio que admita Docker. En Streamlit Community Cloud puede haber límites de recursos y configuración del sistema.
