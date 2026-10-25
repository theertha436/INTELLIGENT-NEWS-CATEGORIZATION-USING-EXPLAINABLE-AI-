from utils import detect_language, translate_to_english


text = """
தமிழ்நாட்டில் அரசு புதிய டிஜிட்டல் கல்வித் திட்டத்தை
அறிமுகப்படுத்தியுள்ளது. இந்த திட்டத்தின் மூலம் கிராமப்புற
பள்ளிகளில் மாணவர்களுக்கு கணினி மற்றும் இணைய வசதிகள்
வழங்கப்படும்.
"""


print("Detected Language:")
print(detect_language(text))

print("\nEnglish Translation:")
print(translate_to_english(text))