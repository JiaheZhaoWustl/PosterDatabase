import os
import json
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv
import httpx
import openai  # pip install openai>=1.14.0
from openai import OpenAI

# python Script_StyleKeywordGen.py ^
#   --input_dir "PosterDataset" ^
#   --output_file "poster_styles.jsonl" ^
#   --model gpt-4o ^
#   --github_base_url "https://raw.githubusercontent.com/JiaheZhaoWustl/PosterDatabase/main/PosterDataset"


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(
    api_key=openai_api_key,
    http_client=httpx.Client(timeout=60.0)
)

# -------------------- 🔧 STYLE-ONLY SYSTEM PROMPT -------------------- #
STYLE_KEYWORD_PROMPT = (
    "You are a graphic design researcher. Given an image of a poster, return a JSON array of 2 to 6 style keywords that best describe the visual language and graphic design aesthetic of the poster.\n\n"
    "Each keyword should be:\n"
    "- Short, lowercase, and specific\n"
    "- Focused on style, technique, layout, texture, or emotional tone\n"
    "- Avoid vague or generic descriptors unless they are uniquely defining\n\n"
    "Examples of acceptable style terms include: \"modernist\", \"collage\", \"riso\", \"digital glitch\", \"surreal\", \"geometric\", \"3D\", \"deconstructivist\", \"neon\", \"retro\", \"playful\", \"halftone\", \"mesh\", \"cinematic\", \"chaotic\", etc.\n\n"
    "If the poster shows a distinctive material, texture, structure, or typographic treatment — such as mesh, kinetic type, layered distortion, halftone, risograph texture, or overprint — include it as a keyword.\n"
    "You may also include conceptual or emotional tones like \"ominous\", \"euphoric\", \"futurist\", \"satirical\", or \"clinical\" if they feel central to the poster’s voice.\n\n"
    "Avoid:\n"
    "- Redundant or default terms like \"bold type\" or \"poster\"\n"
    "- Literal or overly basic palette labels like \"black and white\" unless they define the design style\n"
    "- Forcing a fixed number — only include what feels meaningfully descriptive\n\n"
    "Return only a JSON array. Example:\n"
    "[\"digital glitch\", \"neon\", \"collage\", \"chaotic\"]"
)


# -------------------- NORMALIZATION + FREQUENCY TRACKING -------------------- #

# Normalization map for keyword consistency
NORMALIZE_KEYWORDS = {
    "futuristic": "futurist",
    "bold typography": "bold type",
    "acid": "acidic",
    "glitchy": "digital glitch",
    "collaged": "collage",
    "halftoned": "halftone"
}

keyword_freq = {}  # Tracks keyword usage globally

def normalize_keywords(keywords):
    normalized = []
    for k in keywords:
        k_norm = k.strip().lower()
        k_norm = NORMALIZE_KEYWORDS.get(k_norm, k_norm)
        normalized.append(k_norm)
        keyword_freq[k_norm] = keyword_freq.get(k_norm, 0) + 1
    return sorted(set(normalized))

# -------------------- STYLE-ONLY ANALYSIS FUNCTION -------------------- #
def analyse_poster_styles(image_url: str, model: str, poster_type: str) -> dict:
    messages = [
        {"role": "system", "content": STYLE_KEYWORD_PROMPT},
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
                        f"This poster is part of the '{poster_type}' design style category. "
                        f"Please consider that and generate a JSON array of 2-6 concise style keywords that describe the visual aesthetic of the poster."
                    ),
                },
            ],
        },
    ]

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
    except openai.OpenAIError as e:
        print(f"🛑 OpenAI API error: {e}")
        raise

    content = response.choices[0].message.content.strip()

    # Clean up ```json ... ``` formatting if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)

    try:
        style_keywords = json.loads(content)
        if not isinstance(style_keywords, list):
            raise ValueError("Not a JSON list.")
        return normalize_keywords(style_keywords)
    except Exception:
        print(f"\n⚠️ Invalid style keyword output from GPT:\n{content}\n")
        raise

# -------------------- MAIN SCRIPT -------------------- #
def main():
    parser = argparse.ArgumentParser(description="Batch-generate style keywords from poster images.")
    parser.add_argument("--input_dir", required=True, help="Path to poster image folder")
    parser.add_argument("--output_file", required=True, help="Output .jsonl file")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--github_base_url", required=True, help="Base raw GitHub image URL")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("❌ Please set your OpenAI API key.")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"❌ Folder not found: {input_dir}")

    image_paths = sorted(input_dir.glob("**/*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    # image_paths = image_paths[:1]  # 👈 Limit to 1 image for testing


    if not image_paths:
        print("❗ No images found.")
        return

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for img in image_paths:
            poster_id = img.stem
            poster_type = img.parent.name
            relative_path = img.relative_to(input_dir).as_posix()
            image_url = args.github_base_url.rstrip("/") + "/" + relative_path

            print(f"🔍 {poster_id} ({poster_type}) → {image_url}", end=" … ")

            try:
                # 🔁 Retry logic: generate multiple times and keep the one with rarest keywords
                best_keywords = []
                min_avg_freq = float("inf")
                attempts = 3

                for _ in range(attempts):
                    try:
                        candidate_keywords = analyse_poster_styles(image_url, args.model, poster_type)
                        if not candidate_keywords:
                            continue

                        # Count new vs known keywords
                        new_terms = [k for k in candidate_keywords if k not in keyword_freq]
                        if len(new_terms) == 0:
                            continue  # skip if nothing new at all
                        if len(new_terms) > 2:
                            continue  # too many new ones = likely noise

                        avg_freq = sum(keyword_freq.get(k, 0) for k in candidate_keywords) / len(candidate_keywords)

                        # Choose if rarer and has acceptable new terms
                        if avg_freq < min_avg_freq:
                            best_keywords = candidate_keywords
                            min_avg_freq = avg_freq

                    except Exception as e:
                        print(f"⚠️ Retry failed: {e}")
                        continue

                if not best_keywords:
                    try:
                        # Fallback: use one default result even if no new keywords
                        fallback = analyse_poster_styles(image_url, args.model, poster_type)
                        best_keywords = fallback or []
                        print("🛟 Using fallback result")
                    except Exception as e:
                        print(f"⚠️ Fallback failed: {e}")
                        best_keywords = []

                style_keywords = best_keywords



                result = {
                    "poster_id": poster_id,
                    "poster_type": poster_type,
                    "image_url": image_url,
                    "style_keywords": sorted(set(style_keywords))  # remove duplicates
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print("✓")
            except Exception as e:
                print(f"❌ {e}")

    print(f"\n✅ Done! Style keywords saved to: {args.output_file}")

    print("\n📊 Keyword frequency across posters:")
    for term, count in sorted(keyword_freq.items(), key=lambda x: -x[1]):
        print(f"{term}: {count}")

if __name__ == "__main__":
    main()