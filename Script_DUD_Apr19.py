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
  --output_file "poster_metadata_test_APR21_Ver3.jsonl" ^
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
    "1. 'decided': These are elements typically known or defined before the design process begins. Include standard fields such as:\n"
    "- Title\n"
    "- Date or Date Range\n"
    "- Venue or Organizer\n"
    "- Visual Direction (a list of aesthetic keywords: e.g., 'type-as-image', 'riso', '3D collage', 'psychedelic', 'brutalist')\n\n"
    "You are encouraged to creatively include other fields relevant to the poster's context or genre, such as:\n"
    "- Project Series or Event Theme\n"
    "- Target Audience\n"
    "- Genre or Category (e.g., rave flyer, museum show, student thesis, club event)\n"
    "- Artistic Intent or Conceptual Theme\n"
    "- Cultural Reference (if any)\n\n"
    "Choose only the fields that feel relevant for the specific poster — don't repeat the same structure every time. You are allowed to infer thoughtfully if information is implied but not explicit.\n\n"
    "2. 'undecided': These are visual decisions made during the design process. Include:\n"
    "- Typography Treatment (e.g., warped, modular, calligraphic)\n"
    "- Layout or Composition Logic\n"
    "- Texture or Material Quality (e.g., photocopy, screen blend, vector crispness)\n"
    "- Micro-Information Placement (where dates, locations, or QR codes appear)\n"
    "- Visual Rhythm\n"
    "- Type-Image Interaction\n"
    "- Hierarchy and Focal Flow\n"
    "- Emotional or Conceptual Tone\n\n"
    "You may include any of these or add new subfields based on the specific visual behavior of the poster. Be creative and observational.\n\n"
    "3. 'lva': Layered Visual Analysis. Describe the poster's construction as a sequence of 2-5 visual layers:\n"
    "Layer 1 (Background), Layer 2 (Main Forms), Layer 3 (Typographic Overlays), Layer 4 (Micro-details), etc.\n"
    "Each layer should be a **short paragraph (2-4 sentences)** describing:\n"
    "- What visual material appears in that layer\n"
    "- How it is composed and styled\n"
    "- Its role in the spatial or narrative hierarchy\n"
    "- Its texture, contrast, position, and interaction with other layers\n\n"
    "This section should read like a design breakdown, focusing on how the poster is built and experienced visually.\n\n"
    "🎯 Overall, your job is to combine structured description with interpretive insight — as if preparing archival metadata for a creative research tool. Do not output markdown, code fences, or commentary. Return only valid JSON."
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
            max_tokens=2000, #--------------IMPORTANT-----------------#
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
    image_paths = image_paths[:1]  # 👈 Limit to 1 image for testing

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
