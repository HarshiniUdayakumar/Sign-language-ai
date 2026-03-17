import speech_recognition as sr


def recognize_speech():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening...")

        recognizer.adjust_for_ambient_noise(source)

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        text = text.lower()

        print("Recognized:", text)

        return text

    except sr.UnknownValueError:
        print("Speech not understood")
        return None

    except sr.RequestError:
        print("Speech service error")
        return None

if __name__ == "__main__":
    while True:
        text = recognize_speech()

        if text and "exit" in text:
            print("Stopping...")
            break