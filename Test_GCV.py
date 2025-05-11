import os
import sys
import json
import base64
import requests
import numpy as np

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import io
import cv2

load_dotenv()

def run_google_ocr(image_input, is_url=False, output_img="google_boxed.jpg", output_json="google_ocr_output.json"):
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        raise ValueError("❌ GOOGLE_VISION_API_KEY not set in environment.")

    if is_url:
        image_part = {"source": {"imageUri": image_input}}
    else:
        with open(image_input, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        image_part = {"content": encoded_image}

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    body = {
        "requests": [
            {
                "image": image_part,
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(body))
    result = response.json()

    if "error" in result["responses"][0]:
        print("❌ Error:", result["responses"][0]["error"]["message"])
        return

    annotations = result["responses"][0].get("textAnnotations", [])
    if not annotations:
        print("⚠️ No text found.")
        return

    boxes = []
    for item in annotations[1:]:
        box = [(v.get("x", 0), v.get("y", 0)) for v in item["boundingPoly"]["vertices"]]
        boxes.append({
            "text": item["description"],
            "bbox": box
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(boxes, f, indent=2)
    print(f"📝 OCR result saved to {output_json}")

    # Draw bounding boxes
    if is_url:
        img_data = requests.get(image_input).content
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            draw.polygon(b["bbox"], outline="red", width=2)
        img.save(output_img)
    else:
        img = cv2.imread(image_input)
        for b in boxes:
            pts = [(int(x), int(y)) for x, y in b["bbox"]]
            cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imwrite(output_img, img)

    print(f"📷 Image with bounding boxes saved as {output_img}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path_or_url> [--url]")
        sys.exit(1)

    path_or_url = sys.argv[1]
    use_url = len(sys.argv) > 2 and sys.argv[2] == "--url"
    base_name = os.path.splitext(os.path.basename(path_or_url))[0]
    run_google_ocr(
        path_or_url,
        is_url=use_url,
        output_img=f"{base_name}_google_boxed.jpg",
        output_json=f"{base_name}_google_boxes.json"
    )
