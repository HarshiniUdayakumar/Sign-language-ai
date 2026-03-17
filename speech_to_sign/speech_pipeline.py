from speech_input.speech_recognizer import recognize_speech
from vocabulary.word_filter import filter_word
from communication.unity_sender import send_to_unity


def run_pipeline():

    while True:

        text = recognize_speech()

        if text and "exit" in text:
            print("Stopping system")
            break

        command = filter_word(text)

        if command:
            print("Trigger sign:", command)

            send_to_unity(command)


if __name__ == "__main__":
    run_pipeline()