import streamlit as st
import time
import tempfile
from pathlib import Path
import numpy as np

# Video Library
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor
from realtime_tts import start_tts_worker, pause_tts, resume_tts

# Audio & LLM Library
import whisper
from audiorecorder import audiorecorder
from tts_model import synthesize_speech
from realtime_tts import start_tts_worker
from gemini_module import gemini_get_response

# YOLO
from ultralytics import YOLO

# Result Queue
from queue import Queue

# state variables
if "result_queue" not in st.session_state:
    st.session_state.result_queue = Queue(maxsize=5)

if "is_reading" not in st.session_state:
    st.session_state.is_reading = False

if "skip_reading" not in st.session_state:
    st.session_state.skip_reading = False

if "audio" not in st.session_state:
    st.session_state.audio = None

if "tts_started" not in st.session_state:
    start_tts_worker()
    st.session_state["tts_started"] = True

if "last_sentence" not in st.session_state:
    st.session_state.last_sentence = ""

st.title("VISORA")

# Load YOLO Model
ROOT = Path(__file__).parent
YOLO_WEIGHTS = ROOT / "yolo12n.pt"

@st.cache_resource
def load_yolo():
    model = YOLO(str(YOLO_WEIGHTS))
    names = model.names
    colors = np.random.uniform(0, 255, size=(len(names), 3))
    return model, names, colors

yolo_model, CLASS_NAMES, Colors = load_yolo()

# Speech Interaction
st.subheader("🎤 Voice Interaction")

audio = audiorecorder("Record")

if len(audio) > 0:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")

    with st.spinner("Transcribing..."):
        text = whisper.load_model("base").transcribe(tmp.name)["text"]

    response = gemini_get_response(text)
    out_audio = synthesize_speech(response)
    st.audio(out_audio, autoplay=True)

# Camera Mode
st.subheader("📷 Camera (Detection + OCR)")

col1, col2 = st.columns(2)
with col1:
    start = st.button("📷 Open Camera")
with col2:
    stop = st.button("🛑 Stop Camera")

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start:
    st.session_state.camera_active = True
if stop:
    st.session_state.camera_active = False
    pause_tts()

# ===== CAMERA =====
if st.session_state.camera_active:
    ctx = webrtc_streamer(
        key="visora",
        video_processor_factory=lambda: VideoProcessor(
            yolo_model,
            CLASS_NAMES,
            Colors,
        ),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    if ctx and ctx.state.playing:
        resume_tts()
        st.success("📡 Kamera aktif – deteksi berjalan")
else:
    st.info("Klik **Open Camera** untuk memulai")