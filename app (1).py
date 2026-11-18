import os
import re
import json
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import yt_dlp
from faster_whisper import WhisperModel
from google import genai

st.set_page_config(page_title="ClipForge Studio", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#151a32,#293b68); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.2rem; margin:0;}
.hero p {color:#d8def5; margin:.5rem 0 0 0;}
.stButton>button {border-radius:10px; min-height:3rem; font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎬 ClipForge Studio</h1>
  <p>Convierte videos largos en clips verticales con subtítulos automáticos.</p>
</div>
""", unsafe_allow_html=True)

def run_cmd(args):
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3500:] or "FFmpeg no pudo procesar el video.")
    return proc

def get_video_duration(path):
    result = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(result.stdout.strip())

def download_source(url, folder):
    opts = {
        "outtmpl": str(Path(folder) / "source.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    candidates = list(Path(folder).glob("source.*"))
    candidates = [p for p in candidates if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not candidates:
        raise RuntimeError("No se encontró el video descargado. Comprueba que el enlace sea público y accesible.")
    return max(candidates, key=lambda p: p.stat().st_size)

def transcribe(path, model_size, progress_text):
    progress_text.info("1/4 · Transcribiendo el video…")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(path), beam_size=5, word_timestamps=True, vad_filter=True
    )
    transcript = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start), "end": float(w.end),
                    "text": w.word.strip()
                })
        transcript.append({
            "start": float(seg.start), "end": float(seg.end),
            "text": seg.text.strip(), "words": words
        })
    if not transcript:
        raise RuntimeError("Whisper no detectó voz o no pudo transcribir el audio.")
    return transcript, info

def choose_moments(transcript, duration, count, min_len, max_len, api_key, model_name):
    client = genai.Client(api_key=api_key)
    compact = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        for s in transcript
    )
    prompt = f"""
Eres editor de videos cortos. Selecciona {count} momentos interesantes del video.
Duración total del video: {duration:.1f} segundos.
Cada clip debe durar entre {min_len} y {max_len} segundos.
Busca momentos que tengan sentido por sí solos, con gancho, desarrollo y cierre.
Evita solapamientos entre clips. Usa únicamente tiempos dentro del video.
Devuelve SOLO un JSON válido con esta estructura:
[
  {{"start": 12.0, "end": 48.0, "title": "Título breve"}}
]
Transcripción con marcas de tiempo:
{compact}
"""
    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise RuntimeError("Gemini no devolvió una lista JSON válida. Intenta nuevamente.")
    data = json.loads(match.group(0))
    moments = []
    for item in data:
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
            if min_len <= end - start <= max_len:
                moments.append({
                    "start": start, "end": end,
                    "title": str(item.get("title", "Clip"))[:80]
                })
        except (KeyError, TypeError, ValueError):
            continue
    moments.sort(key=lambda x: x["start"])
    # Remove overlaps and keep requested number.
    clean = []
    for item in moments:
        if not clean or item["start"] >= clean[-1]["end"]:
            clean.append(item)
    if not clean:
        raise RuntimeError("Gemini no encontró clips dentro del rango solicitado. Prueba con otro video o reduce la duración máxima.")
    return clean[:count]

def make_srt(transcript, start, end, output_path):
    words = []
    for seg in transcript:
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg["words"]:
            if w["end"] >= start and w["start"] <= end and w["text"]:
                words.append(w)
    if not words:
        # Fallback to sentence-level captions.
        rows = []
        for seg in transcript:
            a, b = max(seg["start"], start), min(seg["end"], end)
            if b > a and seg["text"]:
                rows.append((a-start, b-start, seg["text"]))
    else:
        rows = []
        group = []
        for w in words:
            group.append(w)
            if len(group) >= 6 or w is words[-1]:
                a = max(start, group[0]["start"]) - start
                b = min(end, group[-1]["end"]) - start
                txt = " ".join(x["text"] for x in group).strip()
                if b > a and txt:
                    rows.append((a, b, txt))
                group = []

    def stamp(sec):
        ms = max(0, int(round(sec * 1000)))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, milli = divmod(rem, 1000)
        return f"{h:02}:{m:02}:{s:02},{milli:03}"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (a, b, txt) in enumerate(rows, 1):
            f.write(f"{i}\n{stamp(a)} --> {stamp(b)}\n{txt}\n\n")

def render_clip(video, transcript, moment, index, outdir):
    start, end = moment["start"], moment["end"]
    duration = end - start
    srt = Path(outdir) / f"clip_{index:02d}.srt"
    make_srt(transcript, start, end, srt)
    output = Path(outdir) / f"clip_{index:02d}.mp4"
    # Scale to fill a 9:16 canvas and center-crop.
    subtitle_path = str(srt).replace("\\", "/").replace(":", r"\:")
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=16,"
        "Bold=1,Outline=2,Shadow=1,Alignment=2,MarginV=100'"
    )
    run_cmd([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video),
        "-t", f"{duration:.3f}", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(output)
    ])
    return output

def process_video(video_path, workdir, api_key, model_name, model_size, count, min_len, max_len, status):
    duration = get_video_duration(video_path)
    if duration < min_len:
        raise RuntimeError(f"El video dura {duration:.0f} s y es más corto que el mínimo de {min_len} s.")
    transcript, info = transcribe(video_path, model_size, status)
    status.info("2/4 · Gemini está seleccionando los mejores momentos…")
    moments = choose_moments(transcript, duration, count, min_len, max_len, api_key, model_name)
    clips_dir = Path(workdir) / "clips"
    clips_dir.mkdir(exist_ok=True)
    outputs = []
    for i, moment in enumerate(moments, 1):
        status.info(f"3/4 · Creando clip {i} de {len(moments)}…")
        outputs.append(render_clip(video_path, transcript, moment, i, clips_dir))
    status.info("4/4 · Preparando el archivo ZIP…")
    zip_path = Path(workdir) / "ClipForge_clips.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outputs:
            z.write(p, arcname=p.name)
    return outputs, zip_path, moments

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Clave API de Gemini", type="password", help="Se usa solo durante esta sesión. No la compartas ni la publiques.")
    model_name = st.text_input("Modelo Gemini", value="gemini-3.8-flash")
    model_size = st.selectbox("Modelo de transcripción", ["small", "medium", "large-v3"], index=0)
    clip_count = st.slider("Cantidad de clips", 1, 10, 5)
    min_len = st.number_input("Duración mínima (segundos)", min_value=20, max_value=180, value=20, step=5)
    max_len = st.number_input("Duración máxima (segundos)", min_value=20, max_value=180, value=180, step=5)
    st.caption("El procesamiento puede tardar bastante según la duración del video y la capacidad del servidor.")

st.subheader("1. Elige el video")
source_type = st.radio("Origen", ["Enlace de YouTube", "Subir archivo"], horizontal=True)
video_url = ""
uploaded = None
if source_type == "Enlace de YouTube":
    video_url = st.text_input("Pega aquí el enlace", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded = st.file_uploader("Selecciona un video", type=["mp4", "mov", "mkv", "webm"])

st.subheader("2. Genera tus clips")
st.caption("Los clips se exportan en formato vertical 9:16 con subtítulos. La duración máxima permitida es de 3 minutos.")
if min_len > max_len:
    st.error("La duración mínima no puede ser mayor que la máxima.")
    st.stop()

if st.button("🚀 Generar clips", type="primary", use_container_width=True):
    if not api_key:
        st.error("Ingresa tu clave API de Gemini en el panel de configuración.")
        st.stop()
    if source_type == "Enlace de YouTube" and not video_url.strip():
        st.error("Pega un enlace de YouTube.")
        st.stop()
    if source_type == "Subir archivo" and uploaded is None:
        st.error("Selecciona un archivo de video.")
        st.stop()

    status = st.empty()
    try:
        with tempfile.TemporaryDirectory(prefix="clipforge_") as tmp:
            tmp_path = Path(tmp)
            if source_type == "Enlace de YouTube":
                status.info("Descargando el video…")
                video_path = download_source(video_url.strip(), tmp_path)
            else:
                video_path = tmp_path / uploaded.name
                with open(video_path, "wb") as f:
                    f.write(uploaded.getbuffer())

            outputs, zip_path, moments = process_video(
                video_path, tmp_path, api_key, model_name.strip() or "gemini-3.8-flash",
                model_size, int(clip_count), int(min_len), int(max_len), status
            )
            st.success(f"¡Listo! Se generaron {len(outputs)} clips.")
            for i, (clip, moment) in enumerate(zip(outputs, moments), 1):
                st.markdown(f"**Clip {i}: {moment['title']}** · {moment['end']-moment['start']:.0f} segundos")
                st.video(str(clip))
            st.download_button(
                "⬇️ Descargar todos los clips (ZIP)",
                data=zip_path.read_bytes(),
                file_name="ClipForge_clips.zip",
                mime="application/zip",
                use_container_width=True
            )
            status.success("Proceso terminado.")
    except Exception as e:
        st.error(f"No se pudo completar el proceso: {e}")
        st.caption("Consejo: verifica la clave de Gemini, el enlace y que el servidor tenga FFmpeg instalado.")

st.divider()
st.caption("Usa videos que tengas derecho a procesar. Algunos directos, videos privados o protegidos pueden no estar disponibles para descarga.")
