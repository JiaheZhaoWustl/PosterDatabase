import easyocr
import cv2
import json
import os
import sys

def detect_and_draw_easyocr(image_path, output_path="boxed_output.jpg", json_output="boxes.json"):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        sys.exit(1)

    reader = easyocr.Reader(['en'])  # You can add 'ch_sim' for Chinese
    print("🔍 Running EasyOCR...")
    results = reader.readtext(image)

    detections = []
    for (bbox, text, conf) in results:
        if conf > 0.7 and text.strip():
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            detections.append({
                "text": text.strip(),
                "bbox": [list(map(int, point)) for point in bbox],
                "confidence": round(conf * 100, 2)
            })

    cv2.imwrite(output_path, image)
    print(f"✅ Saved image with boxes: {output_path}")

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"📝 Saved JSON: {json_output}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Test_EasyOCR_DrawBoxes.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    base = os.path.splitext(os.path.basename(image_path))[0]
    detect_and_draw_easyocr(
        image_path,
        output_path=f"{base}_easyocr_boxed.jpg",
        json_output=f"{base}_easyocr_boxes.json"
    )