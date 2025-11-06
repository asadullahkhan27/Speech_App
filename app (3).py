import os
import streamlit as st
import tempfile
import time
from groq import Groq
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect

# -----------------------------
# STEP 1: API Key setup
# -----------------------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found! Please add it in Streamlit Secrets or Environment Variables.")
    st.stop()

client = Groq(api_key=api_key)

# -----------------------------
# STEP 2: Whisper Transcription
# -----------------------------
def transcribe_audio(audio_path):
    """Transcribe audio using Groq Whisper model."""
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )
        return transcription.text.strip()
    except Exception as e:
        return f"[Transcription error] {e}"

# -----------------------------
# STEP 3: Language Detection
# -----------------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# -----------------------------
# STEP 4: Translation
# -----------------------------
def translate_text(text, dest_lang):
    if not text.strip():
        return "[Translation error] Empty input text"
    try:
        translator = GoogleTranslator(source="auto", target=dest_lang)
        translated = translator.translate(text)
        if translated.strip().lower() == text.strip().lower():
            time.sleep(1)
            translated = translator.translate(text)
        return translated
    except Exception as e:
        return f"[Translation error] {e}"

# -----------------------------
# STEP 5: Text-to-Speech
# -----------------------------
def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        return f"[TTS error] {e}"

# -----------------------------
# STEP 6: Full Pipeline
# -----------------------------
def pipeline_translate(audio_file, target_lang_hint):
    if not audio_file:
        return "[Error] No audio input", None, None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    transcribed = transcribe_audio(tmp_path)
    if transcribed.startswith("[Transcription error]"):
        return transcribed, None, None, None

    detected = detect_language(transcribed)
    if target_lang_hint == "auto":
        target_lang_hint = "ur" if detected == "en" else "en"

    translated = transcribed
    if target_lang_hint != detected:
        translated = translate_text(transcribed, dest_lang=target_lang_hint)

    if translated.startswith("[Translation error]"):
        return transcribed, detected, translated, None

    audio_output = text_to_speech(translated, target_lang_hint)
    if isinstance(audio_output, str) and audio_output.startswith("[TTS error]"):
        return transcribed, detected, translated, None

    return transcribed, detected, translated, audio_output


# -----------------------------
# STEP 7: Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎙 AI Voice Translator", layout="centered")
st.title("🎙 AI Voice Translator")
st.markdown("Speak in any language — it will transcribe, translate, and speak it back!")

audio_file = st.file_uploader("🎤 Upload or Record Your Speech (WAV/MP3)", type=["wav", "mp3"])
target_lang = st.selectbox(
    "🌐 Target Language (auto detects and switches)",
    ["auto", "en", "ur", "fr", "es", "de", "ar"],
    index=0
)

if st.button("🚀 Translate & Speak"):
    with st.spinner("Processing your voice..."):
        transcribed, detected, translated, audio_output = pipeline_translate(audio_file, target_lang)

    if "[Error]" in transcribed or transcribed.startswith("[Transcription error]"):
        st.error(transcribed)
    else:
        st.success("✅ Translation Complete!")
        st.subheader("📝 Transcribed Text")
        st.text(transcribed)

        st.subheader("🌍 Detected Language")
        st.text(detected)

        st.subheader("💬 Translated Text")
        st.text(translated)

        if audio_output and os.path.exists(audio_output):
            st.subheader("🔊 Translated Speech Output")
            audio_file = open(audio_output, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.warning("No audio output generated.")


st.markdown("---")
st.caption("Built with ❤️ using Groq + Streamlit + gTTS + Deep Translator")
