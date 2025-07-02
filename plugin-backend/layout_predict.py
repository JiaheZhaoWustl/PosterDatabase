# layout_predict.py  –  multi-mode + auto-retry edition
# -----------------------------------------------------
#  python layout_predict.py prompt.txt
#  python layout_predict.py prompt.txt --primer primer.jsonl --one-by-one
#  python layout_predict.py poster_prompt.txt --primer primer.jsonl --field title_heat
#
#  Extra CLI options added:
#     --max-retries  (default 4)     how many times to re-ask a field
#     --retry-wait   (default 2.0)   seconds to sleep between retries

import os, sys, json, argparse, textwrap, pathlib, re, numbers, time, random
import numpy as np, matplotlib.pyplot as plt, openai
from openai import APITimeoutError, APIError
from dotenv import load_dotenv
load_dotenv()                         # OPENAI_API_KEY

EXPECTED = [
    "title_heat", "location_heat", "time_heat",
    "host_organization_heat", "call_to_action_purpose_heat",
    "text_descriptions_details_heat",
]

# ─── command-line ────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Call fine-tuned layout model")
ap.add_argument("file", help="Heat-map prompt file or - for STDIN")
ap.add_argument("--primer", help=".jsonl row to prepend as a few-shot example")
ap.add_argument("--model",
                default="ft:gpt-4o-mini-2024-07-18:sia-project-1:jun24-test:Bm0QDLkZ")
ap.add_argument("--one-by-one", action="store_true",
                help="ask the model for one grid per request")
ap.add_argument("--field", choices=EXPECTED,
                help="ask ONLY this field then exit")
ap.add_argument("--temp", type=float, default=1.0,
                help="sampling temperature")
ap.add_argument("--max-retries", type=int, default=10,
                help="max calls per field until we accept zero output")
ap.add_argument("--retry-wait", type=float, default=2.0,
                help="seconds to wait between retries")
ap.add_argument("--no-vis", action="store_true")
args = ap.parse_args()

# ─── helpers ─────────────────────────────────────────────────────────────
def to_grid(val):
    # 1) normalise to list of floats
    if isinstance(val, (list, tuple)):
        nums = [float(x) for x in val]
    else:
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(val))]
    # 2) pad / truncate
    if len(nums) < 252:
        print(f"  · {len(nums)} floats → padded to 252")
        nums += [0.0] * (252 - len(nums))
    elif len(nums) > 252:
        print(f"  · {len(nums)} floats → truncated to 252")
        nums = nums[:252]
    return np.asarray(nums, dtype=float).reshape(21, 12)

def load_primer(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                break
        else:
            sys.exit("❌  primer file is empty")
    msgs = row.get("messages", [])
    ua = [m for m in msgs if m["role"] in ("user", "assistant")]
    if len(ua) >= 2 and ua[0]["role"] == "user" and ua[1]["role"] == "assistant":
        print("✓ primer loaded")
        return ua[:2]
    print("⚠️  primer has no usable user/assistant pair")
    return []

def call_model_once(user_prompt, field=None):
    """Single OpenAI call, may raise exceptions."""
    system_msg = textwrap.dedent(f"""
    <IMAGE_HEAT> Predict {'ONE' if field else 'six'} 12×21 heat-maps.
    Return JSON ONLY{f' with key: {field}' if field else ' with exactly these keys:'}
    {' '.join(EXPECTED) if not field else ''}
    Each value = 252 floats separated by a single space.
    """).strip()

    messages = [{"role": "system", "content": system_msg}] + primer_msgs + [
        {"role": "user", "content": user_prompt if not field
         else f"FIELD {field}\n{user_prompt}"}
    ]

    print(f"→ calling model…{' ('+field+')' if field else ''}")
    resp = openai.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temp,
        response_format={"type": "json_object"},
        max_tokens=2000,
        timeout=40.0
    )
    print("← reply received")
    return json.loads(resp.choices[0].message.content)

def call_model_with_retry(user_prompt, field):
    """Repeat-call until non-zero output or retries exhausted."""
    for attempt in range(1, args.max_retries + 1):
        try:
            data = call_model_once(user_prompt, field=field)
        except (APITimeoutError, APIError) as e:
            print(f"⏳  {field}: API error ({e.__class__.__name__}); retry {attempt}/{args.max_retries}")
            time.sleep(args.retry_wait)
            continue

        if field in data:
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(data[field]))] \
                   if not isinstance(data[field], (list, tuple)) else data[field]
            if any(abs(n) > 1e-6 for n in nums):      # non-zero?
                return data[field]

            print(f"🔁  {field}: all zeros; retry {attempt}/{args.max_retries}")
        else:
            print(f"🔁  {field}: key omitted; retry {attempt}/{args.max_retries}")

        time.sleep(args.retry_wait)

    print(f"⚠️  {field}: still zero/omitted after {args.max_retries} tries → zero-fill")
    return "0.0 " * 252

# ─── read prompt ────────────────────────────────────────────────────────
USER_PROMPT = sys.stdin.read() if args.file == "-" else pathlib.Path(args.file).read_text()
if not USER_PROMPT.strip():
    sys.exit("❌  prompt is empty")

primer_msgs = load_primer(args.primer)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── execution modes ────────────────────────────────────────────────────
if args.field:                        # single field
    val = call_model_with_retry(USER_PROMPT, args.field)
    grid = to_grid(val)
    print(grid)
    sys.exit(0)

if args.one_by_one:                   # field-by-field mode
    reply = {
        fld: call_model_with_retry(USER_PROMPT, fld)
        for fld in EXPECTED
    }
else:                                 # all-at-once mode with retries per missing/zero
    reply_raw = call_model_once(USER_PROMPT)
    reply = {}
    for fld in EXPECTED:
        if fld in reply_raw:
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", str(reply_raw[fld]))]
            if nums and any(abs(n) > 1e-6 for n in nums):
                reply[fld] = reply_raw[fld]
                continue
        # fall back to retry loop if missing or zero
        reply[fld] = call_model_with_retry(USER_PROMPT, fld)

# ─── convert to grids & visualise ───────────────────────────────────────
grids = {k: to_grid(v) for k, v in reply.items()}

if not args.no_vis:
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), dpi=110)
    for i, k in enumerate(EXPECTED):
        r, c = divmod(i, 3)
        ax[r][c].imshow(grids[k], vmin=0, vmax=1, cmap="viridis", origin="upper")
        ax[r][c].set_title(k.replace("_", "\n"), fontsize=7)
        ax[r][c].axis("off")
    plt.tight_layout(); plt.show()
