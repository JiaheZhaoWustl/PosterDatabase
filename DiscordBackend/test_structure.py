import json

# Load a small sample to understand the structure
with open('Midjourney - Newcomer Rooms - newbies-41 [990816772108202044] (after 2025-07-17).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Data type: {type(data)}")
print(f"Data keys: {list(data.keys())}")

if isinstance(data, dict):
    print(f"First key: {list(data.keys())[0]}")
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    print(f"First value type: {type(first_value)}")
    if isinstance(first_value, list):
        print(f"First value length: {len(first_value)}")
        if first_value:
            print(f"First item in list: {first_value[0]}")
    else:
        print(f"First value: {first_value}") 