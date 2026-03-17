SUPPORTED_WORDS = [
    "hello",
    "thankyou",
    "yes",
    "beautiful",
    "vegetable",
    "please",
    "help"
]


def filter_word(text):

    if text is None:
        return None

    text = text.lower()

    for word in SUPPORTED_WORDS:
        if word in text:
            print("Valid command:", word)
            return word

    print("Word not supported")
    return None