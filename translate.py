from deep_translator import GoogleTranslator

# 영어(기본값)를 한국어로 번역
translated = GoogleTranslator(source='auto', target='ko').translate("Hello, how are you?")
print(translated)  # 출력 예: 안녕하세요, 어떻게 지내세요?
