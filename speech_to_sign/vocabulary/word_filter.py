SUPPORTED_WORDS = [
    "hello",
    "thankyou",
    "yes",
    "beautiful",
    "vegetable",
    "please",
    "help",
    "wrong",
    "address"
]


def filter_word(text):

    if text is None:
        return None

    text = text.lower()

    # 🔹 special cases
    if "thank you" in text:
        print("Valid command: thankyou")
        return "thankyou"

    # 🔹 general matching
    for word in SUPPORTED_WORDS:
        if word in text:
            print("Valid command:", word)
            return word

    print("Word not supported")
    return None