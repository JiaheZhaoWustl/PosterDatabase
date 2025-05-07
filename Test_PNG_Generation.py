import os
import requests
from openai import OpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
remove_bg_api_key = os.getenv("REMOVE_BG_API_KEY")

if not openai_api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")
if not remove_bg_api_key:
    raise ValueError("❌ Missing REMOVE_BG_API_KEY in .env")

client = OpenAI(
    api_key=openai_api_key,
    http_client=httpx.Client(timeout=60.0)
)

# Step 2: Ask GPT-4o to refine your DALL·E prompt
base_prompt = (
    "Create a standalone visual image of the numerals '2026' in a vertical 1080x1920px format. "
    "The style should be digital glitch, cyberpunk, and acidic — using neon colors like electric blue, magenta, and neon green. "
    "Each numeral in '2026' should be clearly readable and visually intact — not fragmented into abstract or illegible shapes. "
    "Use glitch textures, distortion, and chaotic overlays only on the numerals themselves. "
    "Design the background to be as flat and solid-colored as possible (preferably white, black, or neutral), so it can be easily removed later to isolate the '2026' as a transparent PNG. "
    "Avoid adding any other elements or text — focus only on the styled '2026'."
)



print("🧠 Asking GPT-4o to refine the prompt...")
try:
    refined_prompt = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a prompt engineer optimizing inputs for DALL·E 3. Make prompts vivid, clear, and layout-aware."},
            {"role": "user", "content": base_prompt}
        ]
    ).choices[0].message.content.strip()
    print(f"🎯 Refined Prompt: {refined_prompt}")
except Exception as e:
    print("❌ Error refining prompt:", e)
    exit(1)

# Step 3: Generate image with DALL·E 3
print("🎨 Generating image from refined prompt...")
try:
    response = client.images.generate(
        model="dall-e-3",
        prompt=refined_prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )
except Exception as e:
    print("❌ Error generating image:", e)
    exit(1)

# Step 4: Download and save the image
image_url = response.data[0].url
raw_path = "poster_raw.png"
try:
    print(f"🌐 Downloading image from: {image_url}")
    image_data = requests.get(image_url).content
    with open(raw_path, "wb") as f:
        f.write(image_data)
    print(f"💾 Image saved as {raw_path}")
except Exception as e:
    print(f"❌ Error downloading image: {e}")
    exit(1)

# Step 5: Remove background using remove.bg API
output_path = "poster_visual_transparent_MAY7.png"
print("🪄 Removing background using remove.bg API...")

try:
    with open(raw_path, "rb") as image_file:
        removebg_response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": image_file},
            data={"size": "auto"},
            headers={"X-Api-Key": remove_bg_api_key},
        )

    if removebg_response.status_code == 200:
        with open(output_path, "wb") as out_file:
            out_file.write(removebg_response.content)
        print(f"✅ Transparent PNG saved as {output_path}")
    else:
        print(f"❌ remove.bg API Error: {removebg_response.status_code} - {removebg_response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Error using remove.bg API: {e}")
    exit(1)
