import pyttsx3

def init_speech_engine():
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    return engine

def announce_detection(engine, label):
    # Convert the detected label into speech
    engine.say(f"{label} detected")
    engine.runAndWait()
