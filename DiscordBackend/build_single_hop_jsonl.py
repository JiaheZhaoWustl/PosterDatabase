# build_single_hop_jsonl.py
# ------------------------------------------------------
#  Needs:
#   pip install openai==1.* pandas rapidfuzz python-dotenv
#   export OPENAI_API_KEY=<your key>
#   python build_single_hop_jsonl.py  \
#          --in  newbies-41_user_prompts_reformatted.json \
#          --out single_hop_gpt4o.jsonl

import argparse
import json
import difflib
import os
import time
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv

load_dotenv()  # Loads OPENAI_API_KEY

# --------------- Configuration -----------------
@dataclass
class Config:
    """Configuration settings for the prompt refinement processor."""
    model_name: str = "gpt-4o-mini"  # or "gpt-4o" if you have access
    max_options: int = 4              # never more than this
    small_edit_threshold: int = 50    # Levenshtein threshold (skip big rewrites)
    max_retries: int = 3
    timeout: int = 40
    temperature: float = 0.7
    
    system_blurb: str = (
        "You are a prompt-refinement assistant. "
        "Ask ONE concise question with no more than four short options. "
        "After the user answers, reply ONLY with 'Updated prompt:' followed by "
        "the improved prompt, and nothing else."
    )

# ----------------------------------------------

class PromptRefinementProcessor:
    """Handles the conversion of prompt pairs to fine-tuning format."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI()  # picks up OPENAI_API_KEY from env
        
        self.question_schema = {
            "name": "make_question",
            "description": "Craft one specific question based on the prompt content type that would help refine the prompt. "
                          "Return 2-4 short, relevant options that match the content type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": config.max_options
                    }
                },
                "required": ["question", "options"]
            }
        }
    
    def diff_added(self, prev: str, curr: str, max_tokens: int = 6) -> str:
        """Return up to max_tokens words added from prev → curr."""
        added = [
            tok[2:] for tok in difflib.ndiff(prev.split(), curr.split())
            if tok.startswith("+ ")
        ]
        return " ".join(added[:max_tokens]) or "no change"
    
    def ask_gpt_question(self, prev: str, added: str) -> Tuple[str, List[str]]:
        """Call GPT to get (question, options) that bridges the gap between original and changes."""
        sys_msg = """You are a helpful AI design assistant helping users refine their Midjourney prompts. Ask natural, conversational questions that help users think through their design choices.

Your questions should sound genuinely interested and collaborative, not directive. Use varied, natural language like:
- "What kind of..." 
- "How do you want to..."
- "Tell me more about..."
- "What's your take on..."
- "I'd like to understand..."
- "What are your thoughts on..."
- "Could you describe..."
- "What style are you going for..."
- "How should we approach..."
- "What feels right for..."
- "What are you thinking for..."
- "What do you have in mind for..."
- "What's your preference for..."
- "What would work best for..."
- "What are you looking for in..."
- "What should we focus on for..."
- "What details matter most for..."
- "What's important to you about..."

Analyze the original prompt and the words the user added. Create a question that, when answered, would naturally result in those added words.

Examples:
- Original: "generate a cat" → Added: "black" → Question: "What kind of color are you thinking for the cat?" Options: "black, white, orange, gray"
- Original: "a chair" → Added: "wooden" → Question: "What material feels right for the chair?" Options: "wooden, metal, plastic, leather"
- Original: "a landscape" → Added: "sunset" → Question: "What time of day works best for this scene?" Options: "sunrise, midday, sunset, night"

The question should be designed so that when the user answers with one of the options, it naturally leads to their change. Make it feel like a natural design conversation, not an interrogation. Avoid repetitive words like 'envision' or 'envisioning' - use varied language instead."""
        user_msg = (
            f"ORIGINAL_PROMPT:\n{prev}\n\n"
            f"WORDS_ADDED:\n{added}\n\n"
            f"Create a question that would naturally lead to the user adding '{added}'. The question should be designed so that when answered, it results in the added words. Use conversational, curious language. Provide 2-4 options where one of them matches the user's change."
        )

        for retry in range(self.config.max_retries):
            try:
                if retry > 0:
                    print(f"      🔄 Retry {retry}/{self.config.max_retries}...")
                
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    tools=[{"type": "function", "function": self.question_schema}],
                    tool_choice={"type": "function", "function": {"name": "make_question"}},
                    temperature=self.config.temperature,
                    timeout=self.config.timeout
                )
                
                args = json.loads(
                    response.choices[0].message.tool_calls[0].function.arguments
                )
                options = args.get("options", [])[:self.config.max_options]
                return args["question"].strip(), options
                
            except (RateLimitError, APIError) as e:
                if retry == self.config.max_retries - 1:
                    print(f"      ❌ API call failed after {self.config.max_retries} retries: {e}")
                else:
                    print(f"      ⏳ Rate limited, waiting {2 ** retry}s...")
                time.sleep(2 ** retry)  # exponential backoff
        
        # fallback
        return "Could you clarify this change?", []
    
    def build_conversation_row(self, prev: str, curr: str) -> Dict:
        """Build a single conversation row for fine-tuning."""
        added = self.diff_added(prev, curr)
        question, options = self.ask_gpt_question(prev, added)
        
        assistant_question = question if not options else f"{question} Options: {', '.join(options)}."
        user_reply = added

        return {
            "messages": [
                {"role": "system", "content": self.config.system_blurb},
                {"role": "user", "content": prev},
                {"role": "assistant", "content": assistant_question},
                {"role": "user", "content": user_reply},
                {"role": "assistant", "content": f"Updated prompt:\n{curr}"}
            ]
        }
    
    def is_valid_pair(self, prev: str, curr: str) -> bool:
        """Check if a prompt pair is valid for processing."""
        if not prev or not curr:  # empty lines
            return False
        if prev == curr:  # exact retry
            return False
        similarity = fuzz.ratio(prev, curr)
        if similarity < (100 - self.config.small_edit_threshold):
            print(f"      📊 Similarity too low: {similarity}% (threshold: {100 - self.config.small_edit_threshold}%)")
            return False  # large rewrite
        return True
    
    def process_sessions(self, sessions: List[Dict], output_path: Path) -> Tuple[int, int]:
        """Process all sessions and write to JSONL file."""
        total_pairs = 0
        kept_pairs = 0
        
        print(f"🔄 Processing {len(sessions)} sessions...")
        
        with output_path.open("w", encoding="utf-8") as fout:
            for session_idx, session in enumerate(sessions, 1):
                prompts = [p["prompt_text"].strip() for p in session["prompts"]]
                session_pairs = len(prompts) - 1
                
                print(f"📝 Session {session_idx}/{len(sessions)}: {session_pairs} prompt pairs")
                
                for pair_idx, (prev, curr) in enumerate(zip(prompts, prompts[1:]), 1):
                    total_pairs += 1
                    
                    if not self.is_valid_pair(prev, curr):
                        print(f"   ⏭️  Skipping pair {pair_idx} (invalid)")
                        print(f"      📝 Prev: {prev[:50]}...")
                        print(f"      📝 Curr: {curr[:50]}...")
                        continue
                    
                    try:
                        print(f"   🤖 Generating question for pair {pair_idx}/{session_pairs}...")
                        row = self.build_conversation_row(prev, curr)
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept_pairs += 1
                        print(f"   ✅ Processed pair {pair_idx}/{session_pairs}")
                    except Exception as e:
                        print(f"   ❌ Failed to process pair {pair_idx}: {e}")
                        continue
        
        return total_pairs, kept_pairs


def create_test_jsonl(output_path: Path = Path("test_single_hop.jsonl")):
    """Create a small test JSONL file for testing."""
    test_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a prompt-refinement assistant. Ask ONE concise question with no more than four short options. After the user answers, reply ONLY with 'Updated prompt:' followed by the improved prompt, and nothing else."},
                {"role": "user", "content": "a cat sitting on a chair"},
                {"role": "assistant", "content": "What style are you looking for? Options: photorealistic, cartoon, minimalist, artistic."},
                {"role": "user", "content": "photorealistic"},
                {"role": "assistant", "content": "Updated prompt:\na photorealistic cat sitting on a wooden chair, natural lighting, detailed fur texture"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a prompt-refinement assistant. Ask ONE concise question with no more than four short options. After the user answers, reply ONLY with 'Updated prompt:' followed by the improved prompt, and nothing else."},
                {"role": "user", "content": "a mountain landscape"},
                {"role": "assistant", "content": "What time of day? Options: sunrise, midday, sunset, night."},
                {"role": "user", "content": "sunset"},
                {"role": "assistant", "content": "Updated prompt:\na mountain landscape at sunset, golden hour lighting, dramatic sky, --ar 16:9"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a prompt-refinement assistant. Ask ONE concise question with no more than four short options. After the user answers, reply ONLY with 'Updated prompt:' followed by the improved prompt, and nothing else."},
                {"role": "user", "content": "a futuristic car"},
                {"role": "assistant", "content": "What perspective? Options: side view, front view, three-quarter view, close-up."},
                {"role": "user", "content": "three-quarter view"},
                {"role": "assistant", "content": "Updated prompt:\na futuristic car in three-quarter view, sleek design, metallic finish, neon lighting, --ar 3:2"}
            ]
        }
    ]
    
    with output_path.open("w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✔ Created test JSONL file: {output_path}")
    return output_path


def main():
    """Main function to process prompt refinement data."""
    parser = argparse.ArgumentParser(description="Convert prompt pairs to fine-tuning format")
    parser.add_argument("--in", dest="input_path", required=True, help="input JSON file")
    parser.add_argument("--out", dest="output_path", default="single_hop_gpt4o.jsonl",
                       help="output fine-tune JSONL")
    parser.add_argument("--test", action="store_true", help="create test JSONL file")
    args = parser.parse_args()
    
    if args.test:
        create_test_jsonl(Path(args.output_path))
        return
    
    config = Config()
    processor = PromptRefinementProcessor(config)
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    try:
        print(f"📖 Loading input file: {input_path}")
        sessions = json.loads(input_path.read_text(encoding="utf-8"))
        print(f"📊 Found {len(sessions)} sessions")
        
        print(f"🚀 Starting processing with {config.model_name}...")
        total_pairs, kept_pairs = processor.process_sessions(sessions, output_path)
        
        print(f"\n🎉 Processing complete!")
        print(f"📈 Statistics:")
        print(f"   • Total prompt pairs: {total_pairs}")
        print(f"   • Valid pairs kept: {kept_pairs}")
        print(f"   • Success rate: {(kept_pairs/total_pairs*100):.1f}%" if total_pairs > 0 else "   • Success rate: 0%")
        print(f"📁 Output saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")


if __name__ == "__main__":
    main()
