import pyttsx3
import threading

def _speak(text):
    engine = pyttsx3.init()

    voices = engine.getProperty("voices")

    # Try female voice if available
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)

    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak(text):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()