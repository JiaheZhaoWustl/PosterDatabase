import easyocr
import sys

reader = easyocr.Reader(['en'])
image_path = sys.argv[1]

results = reader.readtext(image_path)
for (_, text, conf) in results:
    if conf > 0.7:
        print(text.strip())
