import json
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher
from collections import Counter

def extract_user_from_original_content(original_content: str) -> str:
    """
    Extract user name from original_content field.
    Pattern: **prompt** - @Username (fast) or **prompt** - @Username
    """
    if not original_content:
        return None
    
    # Look for @Username pattern
    match = re.search(r'@([^\s]+)', original_content)
    if match:
        return match.group(1)
    return None

def extract_user_from_content(content: str) -> str:
    """
    Extract user name from raw Discord content.
    Pattern: **prompt** - @Username (fast) or similar
    """
    if not content:
        return None
    
    # Look for @Username pattern
    match = re.search(r'@([^\s]+)', content)
    if match:
        return match.group(1)
    return None

def extract_prompt_from_content(content: str) -> str:
    """
    Extract prompt text from raw Discord content.
    Pattern: **prompt text** - @Username (fast)
    """
    if not content:
        return None
    
    # Look for prompt text between ** markers
    match = re.search(r'\*\*([^*]+)\*\*', content)
    if match:
        prompt_text = match.group(1).strip()
        # Remove any trailing parts like " - @Username (fast)"
        prompt_text = re.sub(r'\s*-\s*@[^\s]+\s*\([^)]*\)', '', prompt_text)
        return prompt_text.strip()
    
    return None

def is_new_prompt(action_type: str, message_type: str) -> bool:
    """
    Check if this is a new prompt (not an image variation or selection).
    """
    # New prompts typically have action_type "fast" or "imagine"
    # Image variations have action_type like "image_selection", "variations", etc.
    return action_type in ["fast", "imagine"]

def is_meaningful_prompt(prompt_text: str) -> bool:
    """
    Check if the prompt text is meaningful and not just parameters or empty.
    """
    if not prompt_text:
        return False
    
    # Remove common Midjourney parameters
    cleaned_prompt = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', prompt_text)
    cleaned_prompt = re.sub(r'--[a-zA-Z]+', '', cleaned_prompt)
    cleaned_prompt = cleaned_prompt.strip()
    
    # Check if it's just empty, colon, or very short
    if not cleaned_prompt or cleaned_prompt == ':' or len(cleaned_prompt) < 10:
        return False
    
    # Check if it's just parameters (like ": --v 7.0")
    if re.match(r'^[\s:]*$', cleaned_prompt):
        return False
    
    return True

def clean_prompt_text(prompt_text: str) -> str:
    """
    Clean prompt text by removing image links, URLs, and other unwanted elements.
    """
    if not prompt_text:
        return prompt_text
    
    # Remove image links (common patterns)
    # Remove https://s.mj.run/... links (Midjourney image references)
    prompt_text = re.sub(r'https://s\.mj\.run/[^\s]+', '', prompt_text)
    
    # Remove other image URLs
    prompt_text = re.sub(r'https?://[^\s]+\.(png|jpg|jpeg|gif|webp|svg)', '', prompt_text)
    
    # Remove general URLs
    prompt_text = re.sub(r'https?://[^\s]+', '', prompt_text)
    
    # Remove image references in angle brackets
    prompt_text = re.sub(r'<[^>]*\.(png|jpg|jpeg|gif|webp|svg)[^>]*>', '', prompt_text)
    
    # Remove image references in square brackets
    prompt_text = re.sub(r'\[[^\]]*\.(png|jpg|jpeg|gif|webp|svg)[^\]]*\]', '', prompt_text)
    
    # Clean up extra whitespace
    prompt_text = re.sub(r'\s+', ' ', prompt_text)
    prompt_text = prompt_text.strip()
    
    return prompt_text

def is_english_prompt(prompt_text: str) -> bool:
    """
    Check if the prompt is primarily in English.
    For now, accept all prompts to avoid filtering out valid content.
    """
    return True  # Accept all prompts for now

def is_meaningful_prompt_for_ml(prompt_text: str) -> bool:
    """
    Check if prompt is meaningful enough for machine learning.
    Filters out very short, repetitive, or low-quality prompts.
    """
    # Clean the prompt first
    cleaned = clean_prompt_text(prompt_text)
    
    # Skip if empty after cleaning
    if not cleaned.strip():
        return False
    
    # Count words (excluding parameters)
    words = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', cleaned)
    words = re.sub(r'--[a-zA-Z]+', '', words)
    words = words.strip().split()
    
    # Filter out very short prompts (less than 5 words)
    if len(words) < 5:
        return False
    
    # Filter out prompts that are just a few words with commas
    if len(words) <= 6 and ',' in cleaned:
        # Check if it's just a list of words
        word_list = [w.strip() for w in cleaned.split(',')]
        if len(word_list) <= 3:  # Just 2-3 comma-separated items
            return False
    
    # Filter out prompts that are just file paths or system commands
    if cleaned.startswith('C:\\') or cleaned.startswith('/') or cleaned.startswith('<'):
        return False
    
    # Filter out prompts that are just model parameters or commands
    command_keywords = ['change model', 'model <', '< <']
    if any(cmd in cleaned.lower() for cmd in command_keywords):
        return False
    
    return True

def clean_raw_discord_data(input_file: str, output_file: str):
    """
    Clean raw Discord data to extract only user names and their original text prompts.
    This handles the raw Discord message structure.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract messages from the Discord export structure
    messages = data.get('messages', [])
    if not messages:
        print("No messages found in the file")
        return
    
    cleaned_data = []
    seen_prompts = set()  # To track duplicates
    
    for entry in messages:
        # Extract content from raw Discord message
        content = entry.get('content', '')
        if not content:
            continue
        
        # Extract user name from content (look for @Username pattern)
        user_name = extract_user_from_content(content)
        if not user_name:
            continue
        
        # Extract prompt text from content
        prompt_text = extract_prompt_from_content(content)
        if not prompt_text:
            continue
        
        # Clean the prompt text (remove image links, URLs, etc.)
        cleaned_prompt_text = clean_prompt_text(prompt_text)
        
        # Skip if prompt is empty after cleaning
        if not cleaned_prompt_text:
            continue
        
        # Check if prompt is meaningful
        if not is_meaningful_prompt(cleaned_prompt_text):
            continue
        
        # Check if prompt is meaningful for ML (filters out very short/low-quality prompts)
        if not is_meaningful_prompt_for_ml(cleaned_prompt_text):
            continue
        
        # Check if prompt is in English
        if not is_english_prompt(cleaned_prompt_text):
            continue
        
        # Check for duplicates (case-insensitive)
        prompt_lower = cleaned_prompt_text.lower().strip()
        if prompt_lower in seen_prompts:
            continue
        seen_prompts.add(prompt_lower)
        
        # Create cleaned entry
        cleaned_entry = {
            'user_name': user_name,
            'prompt_text': cleaned_prompt_text,
            'timestamp': entry.get('timestamp', ''),
            'action_type': 'fast'  # Default for raw data
        }
        
        cleaned_data.append(cleaned_entry)
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned raw Discord data saved to {output_file}")
    print(f"Total entries processed: {len(data)}")
    print(f"Cleaned entries: {len(cleaned_data)}")
    
    # Print unique users
    unique_users = set(entry['user_name'] for entry in cleaned_data)
    print(f"Unique users: {len(unique_users)}")
    print("Users:", sorted(unique_users))
    
    # Print some statistics
    print(f"\nFiltering statistics:")
    print(f"- Meaningful prompts: {len([e for e in cleaned_data if is_meaningful_prompt(e['prompt_text'])])}")
    print(f"- English prompts: {len([e for e in cleaned_data if is_english_prompt(e['prompt_text'])])}")
    print(f"- Unique prompts (no duplicates): {len(cleaned_data)}")

def clean_discord_data(input_file: str, output_file: str):
    """
    Clean Discord data to extract only user names and their original text prompts.
    This handles the already cleaned data structure.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = []
    seen_prompts = set()  # To track duplicates
    
    for entry in data:
        # Only process entries that are new prompts (not image variations)
        if not is_new_prompt(entry.get('action_type', ''), entry.get('message_type', '')):
            continue
        
        # Extract user name from original_content
        user_name = extract_user_from_original_content(entry.get('original_content', ''))
        if not user_name:
            continue
        
        # Get the prompt text
        prompt_text = entry.get('prompt_text', '')
        if not prompt_text:
            continue
        
        # Clean the prompt text (remove image links, URLs, etc.)
        cleaned_prompt_text = clean_prompt_text(prompt_text)
        
        # Skip if prompt is empty after cleaning
        if not cleaned_prompt_text:
            continue
        
        # Check if prompt is meaningful
        if not is_meaningful_prompt(cleaned_prompt_text):
            continue
        
        # Check if prompt is meaningful for ML (filters out very short/low-quality prompts)
        if not is_meaningful_prompt_for_ml(cleaned_prompt_text):
            continue
        
        # Check if prompt is in English
        if not is_english_prompt(cleaned_prompt_text):
            continue
        
        # Check for duplicates (case-insensitive)
        prompt_lower = cleaned_prompt_text.lower().strip()
        if prompt_lower in seen_prompts:
            continue
        seen_prompts.add(prompt_lower)
        
        # Create cleaned entry
        cleaned_entry = {
            'user_name': user_name,
            'prompt_text': cleaned_prompt_text,
            'timestamp': entry.get('timestamp', ''),
            'action_type': entry.get('action_type', '')
        }
        
        cleaned_data.append(cleaned_entry)
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned data saved to {output_file}")
    print(f"Total entries processed: {len(data)}")
    print(f"Cleaned entries: {len(cleaned_data)}")
    
    # Print unique users
    unique_users = set(entry['user_name'] for entry in cleaned_data)
    print(f"Unique users: {len(unique_users)}")
    print("Users:", sorted(unique_users))
    
    # Print some statistics
    print(f"\nFiltering statistics:")
    print(f"- Meaningful prompts: {len([e for e in cleaned_data if is_meaningful_prompt(e['prompt_text'])])}")
    print(f"- English prompts: {len([e for e in cleaned_data if is_english_prompt(e['prompt_text'])])}")
    print(f"- Unique prompts (no duplicates): {len(cleaned_data)}")

def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings with improved thematic detection"""
    # First clean the prompts to remove image links and URLs
    a = clean_prompt_text(a)
    b = clean_prompt_text(b)
    
    # Clean prompts by removing parameters and special characters
    a_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', a.lower())
    a_clean = re.sub(r'--[a-zA-Z]+', '', a_clean)
    a_clean = re.sub(r'[^\w\s]', ' ', a_clean)
    
    b_clean = re.sub(r'--[a-zA-Z]+\s+[^\s]+', '', b.lower())
    b_clean = re.sub(r'--[a-zA-Z]+', '', b_clean)
    b_clean = re.sub(r'[^\w\s]', ' ', b_clean)
    
    # Calculate basic similarity
    basic_similarity = SequenceMatcher(None, a_clean, b_clean).ratio()
    
    # Get word sets
    a_words = set(a_clean.split())
    b_words = set(b_clean.split())
    
    # Calculate word overlap similarity
    if a_words and b_words:
        word_overlap = len(a_words.intersection(b_words))
        word_union = len(a_words.union(b_words))
        word_similarity = word_overlap / word_union if word_union > 0 else 0
    else:
        word_similarity = 0
    
    # Calculate thematic similarity using word frequency analysis
    # Instead of hard-coded keywords, use the most frequent words in each prompt
    a_word_freq = Counter(a_words)
    b_word_freq = Counter(b_words)
    
    # Get top 5 most frequent words from each prompt
    a_top_words = set(word for word, _ in a_word_freq.most_common(5))
    b_top_words = set(word for word, _ in b_word_freq.most_common(5))
    
    # Calculate overlap of top words
    if a_top_words and b_top_words:
        top_word_overlap = len(a_top_words.intersection(b_top_words))
        max_top_words = max(len(a_top_words), len(b_top_words))
        thematic_similarity = top_word_overlap / max_top_words if max_top_words > 0 else 0
    else:
        thematic_similarity = 0
    
    # Combine similarities with weights
    # For longer prompts, give more weight to thematic similarity
    if len(a_clean.split()) > 10 and len(b_clean.split()) > 10:
        combined_similarity = (basic_similarity * 0.2) + (word_similarity * 0.3) + (thematic_similarity * 0.5)
    else:
        # For shorter prompts, give more weight to basic similarity
        combined_similarity = (basic_similarity * 0.5) + (word_similarity * 0.3) + (thematic_similarity * 0.2)
    
    # Special handling for very short prompts (like Patrick's Portuguese prompts)
    # If both prompts are short and share common words, boost similarity
    if len(a_clean.split()) <= 8 and len(b_clean.split()) <= 8:
        common_words = a_words.intersection(b_words)
        if len(common_words) >= 2:  # If they share at least 2 words
            combined_similarity = max(combined_similarity, 0.6)  # Boost to at least 0.6
    
    # Special handling for longer prompts - boost similarity if they share common words
    # This helps group related prompts without hardcoded keywords
    if len(a_clean.split()) > 15 and len(b_clean.split()) > 15:
        common_words = a_words.intersection(b_words)
        if len(common_words) >= 3:  # If they share at least 3 words
            combined_similarity = max(combined_similarity, 0.4)  # Boost to at least 0.4
    
    return combined_similarity

def extract_changes(prev_prompt: str, current_prompt: str) -> str:
    """Extract the changes between two prompts"""
    if not prev_prompt:
        return "Initial prompt"
    
    # Simple word-level comparison
    prev_words = set(prev_prompt.lower().split())
    current_words = set(current_prompt.lower().split())
    
    added = current_words - prev_words
    removed = prev_words - current_words
    
    changes = []
    if added:
        changes.append(f"Added: {', '.join(sorted(added))}")
    if removed:
        changes.append(f"Removed: {', '.join(sorted(removed))}")
    
    if not changes:
        return "Minor modifications"
    
    return " | ".join(changes)

def group_prompts_by_user_and_session(data: List[Dict[str, Any]], similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    Group prompts by user and session (when prompts become significantly different).
    
    Args:
        data: List of prompt entries with user_name, prompt_text, timestamp
        similarity_threshold: Threshold for considering prompts as part of same session
    
    Returns:
        List of user sessions with grouped prompts
    """
    # First, group all prompts by user to ensure we only compare within the same user
    user_prompts = {}
    for entry in data:
        user_name = entry['user_name']
        if user_name not in user_prompts:
            user_prompts[user_name] = []
        user_prompts[user_name].append(entry)
    
    # Sort each user's prompts by timestamp
    for user_name in user_prompts:
        user_prompts[user_name].sort(key=lambda x: x['timestamp'])
    
    user_sessions = {}
    
    # Process each user's prompts separately
    for user_name, user_data in user_prompts.items():
        user_sessions[user_name] = []
        
        for entry in user_data:
            prompt_text = entry['prompt_text']
            timestamp = entry['timestamp']
            
            # Check if this is a new session (significantly different from last prompt)
            is_new_session = True
            if user_sessions[user_name]:
                last_session = user_sessions[user_name][-1]
                if last_session['prompts']:
                    last_prompt = last_session['prompts'][-1]['prompt_text']
                    similarity = similarity_ratio(last_prompt, prompt_text)
                    # If similarity is high, it's likely the same session
                    if similarity > similarity_threshold:
                        is_new_session = False
            
            if is_new_session:
                # Start new session
                user_sessions[user_name].append({
                    'session_id': len(user_sessions[user_name]) + 1,
                    'prompts': []
                })
            
            # Add prompt to current session
            current_session = user_sessions[user_name][-1]
            
            # Calculate changes from previous prompt
            prev_prompt = None
            if current_session['prompts']:
                prev_prompt = current_session['prompts'][-1]['prompt_text']
            
            changes = extract_changes(prev_prompt, prompt_text)
            
            current_session['prompts'].append({
                'prompt_text': prompt_text,
                'timestamp': timestamp,
                'changes': changes
            })
    
    # Convert to final format
    result = []
    for user_name, sessions in user_sessions.items():
        for session in sessions:
            result.append({
                'user_name': user_name,
                'session_id': session['session_id'],
                'prompts': session['prompts']
            })
    
    return result

def reformat_prompts_by_session(input_file: str, output_file: str, similarity_threshold: float = 0.2):
    """
    Reformat prompts to group by user sessions and show prompt evolution.
    Filters out sessions with only one prompt as they're not useful for ML.
    
    Args:
        input_file: Path to the cleaned prompts JSON file
        output_file: Path to save the reformatted JSON
        similarity_threshold: Threshold for considering prompts as part of same session
    """
    # Read the cleaned data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort by timestamp to ensure chronological order
    data.sort(key=lambda x: x['timestamp'])
    
    # Group prompts by user and session
    reformatted_data = group_prompts_by_user_and_session(data, similarity_threshold)
    
    # Filter out sessions with only one prompt (not useful for ML)
    filtered_data = []
    single_prompt_sessions = 0
    
    for session in reformatted_data:
        if len(session['prompts']) > 1:
            filtered_data.append(session)
        else:
            single_prompt_sessions += 1
    
    # Write the reformatted JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(data)} prompts into {len(reformatted_data)} user sessions")
    print(f"Filtered out {single_prompt_sessions} single-prompt sessions")
    print(f"Final dataset: {len(filtered_data)} multi-prompt sessions")
    print(f"Reformatted data saved to '{output_file}'")
    
    # Print some statistics
    total_prompts = sum(len(session['prompts']) for session in filtered_data)
    avg_prompts_per_session = total_prompts / len(filtered_data) if filtered_data else 0
    
    print(f"\nReformatting statistics:")
    print(f"- Total prompts: {total_prompts}")
    print(f"- Multi-prompt sessions: {len(filtered_data)}")
    print(f"- Single-prompt sessions removed: {single_prompt_sessions}")
    print(f"- Average prompts per session: {avg_prompts_per_session:.1f}")
    
    # Show some examples
    print(f"\nExample sessions:")
    for i, session in enumerate(filtered_data[:3]):
        print(f"  {session['user_name']} (Session {session['session_id']}): {len(session['prompts'])} prompts")

if __name__ == "__main__":
    # Process newbies-41 channel data (raw Discord data)
    input_file = "Midjourney - Newcomer Rooms - newbies-41 [990816772108202044] (after 2025-07-17).json"
    output_file = "newbies-41_user_prompts_only.json"
    clean_raw_discord_data(input_file, output_file)
    
    # Also create the reformatted version
    reformat_prompts_by_session(output_file, "newbies-41_user_prompts_reformatted.json") 