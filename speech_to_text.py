import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

# Load the Whisper model
model = whisper.load_model("base")

st.title("Live Audio Transcription")

# Initialize session state variables
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "audio_data" not in st.session_state:
    st.session_state.audio_data = []
if "recording_event" not in st.session_state:
    st.session_state.recording_event = threading.Event()

# Function to record audio
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)  # Debug: Print status if there's an issue
    st.session_state.audio_queue.put(indata.copy())
    print("Audio chunk added to queue", flush=True)  # Debug: Print when data is added to the queue

def start_recording():
    st.session_state.recording_event.set()
    st.session_state.audio_data = []
    print("Recording started", flush=True)  # Debug: Print when recording starts
    with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
        while st.session_state.recording_event.is_set():
            sd.sleep(100)
    print("Recording stopped", flush=True)  # Debug: Print when recording stops

def stop_recording():
    st.session_state.recording_event.clear()
    audio_chunks = []
    while not st.session_state.audio_queue.empty():
        audio_chunk = st.session_state.audio_queue.get()
        audio_chunks.append(audio_chunk)
        print(f"Audio chunk size: {audio_chunk.shape}", flush=True)  # Debug: Print size of each chunk
    if audio_chunks:
        audio = np.concatenate(audio_chunks, axis=0)
        print(f"Total audio size: {audio.shape}", flush=True)  # Debug: Print total audio size
        transcription = transcribe_audio(audio)
        st.write("Transcription:")
        st.write(transcription)
    else:
        st.write("No audio data recorded.")
        print("No audio data recorded.", flush=True)  # Debug: Print if no audio data is recorded

# Function to transcribe audio
def transcribe_audio(audio):
    audio = audio.flatten()
    result = model.transcribe(audio, fp16=False)
    return result["text"]

# Start/Stop recording button
if st.button("Start/Stop Recording"):
    if not st.session_state.recording:
        st.session_state.recording = True
        st.write("Recording... Click the button again to stop.")
        threading.Thread(target=start_recording).start()
    else:
        st.session_state.recording = False
        stop_recording()
