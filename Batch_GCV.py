import os
from pathlib import Path
from Test_GCV import run_google_ocr

# Paths
input_root = Path("E:/SIA_works/PosterDatabase/PosterDataset")
output_root_img = Path("E:/SIA_works/PosterDatabase/outputs/boxed")
output_root_json = Path("E:/SIA_works/PosterDatabase/outputs/jsonl")

# Create output directories
output_root_img.mkdir(parents=True, exist_ok=True)
output_root_json.mkdir(parents=True, exist_ok=True)

processed_count = 0

for style_folder in input_root.iterdir():
    if not style_folder.is_dir():
        continue

    style_name = style_folder.name

    for img_file in style_folder.iterdir():
        if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        base_name = img_file.stem
        output_img = output_root_img / f"{style_name}_{base_name}_boxed.jpg"
        output_json = output_root_json / f"{style_name}_{base_name}.jsonl"

        print(f"▶ Processing: {img_file}")
        try:
            run_google_ocr(
                image_input=str(img_file),
                is_url=False,
                output_img=str(output_img),
                output_json=str(output_json)
            )
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed on {img_file.name}: {e}")

print(f"\n✅ Finished processing {processed_count} images.")
