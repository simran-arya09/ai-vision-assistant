import pyttsx3
import threading
import queue

engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

speech_queue = queue.Queue()

def worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

thread = threading.Thread(target=worker, daemon=True)
thread.start()

def speak(text):
    if speech_queue.empty():
        speech_queue.put(text)