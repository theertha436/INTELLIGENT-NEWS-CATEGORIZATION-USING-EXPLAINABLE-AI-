from utils import (
    detect_language,
    translate_to_english
)


text = """
இந்திய அணி ஆஸ்திரேலியாவை வீழ்த்தி
சிறப்பான வெற்றியை பெற்றது.
"""


language = detect_language(text)

print(
    "Detected Language:",
    language
)


translated = translate_to_english(text)

print(
    "English Translation:",
    translated
)