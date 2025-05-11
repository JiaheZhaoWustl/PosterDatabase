"""
Generate Decided-Undecided (D-UD) information and Layered Visual Analysis (LVA)
for a batch of poster images using the OpenAI API (GPT-4o).

Usage:
  # 1. Add your OpenAI key to the environment first
  export OPENAI_API_KEY="sk-..."

  # 2. Run the script, pointing at a folder of .png / .jpg posters
  python generate_poster_metadata.py \
      --input_dir ./posters \
      --output_file poster_metadata.jsonl \
      --model gpt-4o-mini            # or gpt-4o, gpt-4o-vision-preview, etc.

The script will create/append to <output_file>, writing one JSONL line per poster
with the following structure:
  {
    "poster_id": "TAI_042",          # filename stem
    "decided": {
        "Title": "…",               # key-value pairs
        ...
    },
    "undecided": {
        "Typography Treatment": "…",
        ...
    },
    "lva": {
        "Layer 1": "…",
        "Layer 2": "…",
        ...
    }
  }
"""

#To fine tune in cmd
""" 
python Script_DUD_Apr19.py ^
  --input_dir "PosterDataset" ^
  --output_file "poster_metadata_batch_APR24.jsonl" ^
  --model gpt-4o ^
  --github_base_url "https://raw.githubusercontent.com/JiaheZhaoWustl/PosterDatabase/main/PosterDataset"
"""

import os
import json
import re
import argparse
from pathlib import Path

import openai  # pip install openai>=1.14.0, currently 1.68.2

# -------------------- 🔧 SYSTEM PROMPT -------------------- #
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior graphic design analyst. For each poster image, return a rich JSON object with three top-level keys:\n\n"
    "1. 'decided': Describe pre-design content that would be known before layout begins. Include common fields like:\n"
    "- Title\n"
    "- Date or Date Range\n"
    "- Venue or Organizer\n"
    "- Visual Direction (keywords: e.g., 'type-as-image', '3D collage', 'black and white')\n"
    "- And optionally: Genre, Target Audience, Artistic Intent, Cultural Reference, or Project Series\n\n"
    "Only use fields that apply to this poster — do not force generic structure. Think like a researcher or creative director preparing design metadata.\n\n"
    "2. 'undecided': Describe visual and compositional decisions made during design. Include fields such as:\n"
    "- Typography Treatment\n"
    "- Layout Logic\n"
    "- Texture / Materiality\n"
    "- Visual Rhythm\n"
    "- Type-Image Interaction\n"
    "- Hierarchy and Focal Flow\n"
    "- Micro-Information Placement\n"
    "- Emotional or Conceptual Tone\n"
    "- Spatial Compression or Dimensional Illusion\n\n"
    "Describe what the designer was visually experimenting with. Be creative and observational.\n\n"
    "3. 'lva': Layered Visual Analysis. Break the poster into 2-5 distinct layers:\n"
    "Each layer should be a short paragraph (2-4 sentences) describing:\n"
    "- What's in the layer (shapes, images, text, symbols)\n"
    "- How it behaves visually (overlapping, floating, compressed, tilted, etc.)\n"
    "- What material or tone it expresses (grainy, glossy, digital)\n"
    "- Its spatial relationship to other layers (foreground, middle, etc.)\n"
    "- Capture the energy and rhythm of the design — not just technical structure\n\n"
    "Example phrasing: 'Tilted white cubes appear to tumble diagonally across the page, creating visual rhythm and tension.'\n\n"
    "Your job is to document and interpret the visual construction of the poster clearly and creatively. Do not use markdown or commentary. Return only valid JSON."
)



# -------------------- POSTER ANALYSIS FUNCTION -------------------- #
def analyse_poster(image_url: str, model: str, system_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": (
                        "Please analyze this poster image and return a JSON object with the following fields: "
                        "'decided' (pre-design info), 'undecided' (visual execution), and 'lva' (Layered Visual Analysis). "
                        "Return JSON only — no explanations."
                    ),
                },
            ],
        },
    ]

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=2500, #--------------IMPORTANT-----------------#
        )
    except openai.OpenAIError as e:
        print(f"🛑 OpenAI API error: {e}")
        raise

    content = response.choices[0].message.content.strip()

    # Clean up ```json ... ``` formatting if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"\n Invalid or malformed JSON from GPT:\n{content}\n")
        raise

# -------------------- MAIN SCRIPT -------------------- #
def main():
    parser = argparse.ArgumentParser(description="Batch-analyze poster images using GPT-4o + GitHub URLs")
    parser.add_argument("--input_dir", required=True, help="Path to local poster folder (e.g., './Poster Dataset/3D')")
    parser.add_argument("--output_file", required=True, help="Output .jsonl file path")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--github_base_url", required=True, help="Base raw GitHub URL")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt (optional)")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("❌ Please set your OpenAI API key: export OPENAI_API_KEY='sk-...'")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"❌ Input directory not found: {input_dir}")

    image_paths = sorted(input_dir.glob("**/*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    # image_paths = image_paths[:1]  # 👈 Limit to 1 image for testing

    if not image_paths:
        print("❗ No image files found.")
        return

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for img in image_paths:
            poster_id = img.stem
            poster_type = img.parent.name
            relative_path = img.relative_to(input_dir).as_posix()
            image_url = args.github_base_url.rstrip("/") + "/" + relative_path

            print(f"🔍 Analyzing {poster_id} ({poster_type}) → {image_url} …", end=" ", flush=True)

            try:
                result = analyse_poster(image_url, args.model, args.system_prompt)
                result["poster_id"] = poster_id
                result["poster_type"] = poster_type
                result["image_url"] = image_url

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print("✓")
            except Exception as e:
                print(f"❌ ERROR: {e}")

    print(f"\n✅ All done! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
