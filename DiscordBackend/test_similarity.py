import re
from difflib import SequenceMatcher

def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings with improved thematic detection"""
    # Clean prompts by removing parameters
    a_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', a.lower())
    a_clean = re.sub(r'--[a-zA-Z]+', '', a_clean)
    b_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', b.lower())
    b_clean = re.sub(r'--[a-zA-Z]+', '', b_clean)
    
    # Calculate basic similarity
    basic_similarity = SequenceMatcher(None, a_clean, b_clean).ratio()
    
    # Also check for thematic similarity (key words)
    a_words = set(a_clean.split())
    b_words = set(b_clean.split())
    
    # Check for thematic keywords that indicate same session
    thematic_keywords = [
        'retro', 'pixel', 'art', 'style', 'illustration', 'design',
        'formula', 'f1', 'race', 'car', 'brazilian', 'senna',
        'streetwear', 't-shirt', 'print', 'vibrant', 'colors'
    ]
    
    a_thematic = a_words.intersection(set(thematic_keywords))
    b_thematic = b_words.intersection(set(thematic_keywords))
    
    thematic_overlap = len(a_thematic.intersection(b_thematic))
    max_thematic = max(len(a_thematic), len(b_thematic))
    
    thematic_similarity = thematic_overlap / max_thematic if max_thematic > 0 else 0
    
    # Combine basic similarity with thematic similarity
    # Give more weight to thematic similarity for better session detection
    combined_similarity = (basic_similarity * 0.3) + (thematic_similarity * 0.7)
    
    return combined_similarity

# Patrick's first two prompts
prompt1 = "retro video game style artwork inspired by Top Gear, Formula 1 race car driving at high speed, pixel art aesthetic, Ayrton Senna tribute, Brazilian flag in the background, tropical landscape with palm trees and Christ the Redeemer statue, vibrant 16-bit color palette (green, yellow, blue), sunset sky gradient, speed HUD and motion lines, nostalgic and heroic atmosphere, bold text overlay: \"Senna. O Brasil em alta velocidade.\" --v 6.0 --ar 3:2 --raw"

prompt2 = "retro pixel art style illustration for streetwear t-shirt design, Formula 1 car with Brazilian theme, Ayrton Senna's yellow helmet with green stripe, bold and clean 16-bit game style, no background, isolated design, vibrant retro colors (yellow, green, blue), stylized for print, includes street-style text: \"O Brasil nunca freia.\" --v 6.0 --ar 1:1 --no background --raw"

similarity = similarity_ratio(prompt1, prompt2)
print(f"Similarity between Patrick's first two prompts: {similarity:.3f}")

# Clean the prompts to see what we're comparing
a_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', prompt1.lower())
a_clean = re.sub(r'--[a-zA-Z]+', '', a_clean)
b_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', prompt2.lower())
b_clean = re.sub(r'--[a-zA-Z]+', '', b_clean)

print(f"\nCleaned prompt 1: {a_clean}")
print(f"Cleaned prompt 2: {b_clean}")

# Check thematic keywords
thematic_keywords = [
    'retro', 'pixel', 'art', 'style', 'illustration', 'design',
    'formula', 'f1', 'race', 'car', 'brazilian', 'senna',
    'streetwear', 't-shirt', 'print', 'vibrant', 'colors'
]

a_words = set(a_clean.split())
b_words = set(b_clean.split())

a_thematic = a_words.intersection(set(thematic_keywords))
b_thematic = b_words.intersection(set(thematic_keywords))

print(f"\nThematic keywords in prompt 1: {a_thematic}")
print(f"Thematic keywords in prompt 2: {b_thematic}")
print(f"Thematic overlap: {a_thematic.intersection(b_thematic)}")

# Calculate individual components
basic_similarity = SequenceMatcher(None, a_clean, b_clean).ratio()
thematic_overlap = len(a_thematic.intersection(b_thematic))
max_thematic = max(len(a_thematic), len(b_thematic))
thematic_similarity = thematic_overlap / max_thematic if max_thematic > 0 else 0

print(f"\nBasic similarity: {basic_similarity:.3f}")
print(f"Thematic similarity: {thematic_similarity:.3f}")
print(f"Combined similarity: {similarity:.3f}")