# Discord Backend

This directory contains tools for processing Discord data, specifically for cleaning and extracting user prompts from Midjourney Discord channels.

## Main Script: `discord_data_cleaner.py`

A flexible command-line tool for cleaning Discord data and extracting user prompts for training datasets.

### Features

- **Flexible Input**: Process any Discord export file or already cleaned data
- **Multiple Output Formats**: Generate cleaned data and/or session-based reformatted data
- **Configurable Parameters**: Adjust similarity thresholds and processing options
- **Comprehensive Filtering**: Remove duplicates, low-quality prompts, and non-English content
- **Session Grouping**: Group related prompts by user and similarity

### Usage

#### Basic Usage
```bash
# Process raw Discord export
python discord_data_cleaner.py my_discord_export.json

# Process with custom output file
python discord_data_cleaner.py my_discord_export.json -o cleaned_data.json
```

#### Advanced Usage
```bash
# Create both cleaned and reformatted output
python discord_data_cleaner.py my_discord_export.json -r

# Custom similarity threshold for session grouping
python discord_data_cleaner.py my_discord_export.json -r -s 0.3

# Process already cleaned data
python discord_data_cleaner.py cleaned_data.json -t cleaned

# Full example with all options
python discord_data_cleaner.py my_discord_export.json -o cleaned.json -r -ro reformatted.json -s 0.25
```

### Command Line Options

- `input_file`: Path to the input Discord data file (required)
- `--output, -o`: Output file path (default: input_file_cleaned.json)
- `--reformat, -r`: Also create reformatted session-based output
- `--reformat-output`: Reformatted output file path (default: input_file_reformatted.json)
- `--similarity-threshold, -s`: Similarity threshold for session grouping (default: 0.2)
- `--data-type, -t`: Type of input data: raw Discord export or already cleaned data (default: raw)
- `--examples`: Show detailed usage examples

### Data Types

#### Raw Discord Data
- Format: Discord export JSON structure
- Contains: Raw message content with user mentions and timestamps
- Processing: Extracts user names and prompts from message content

#### Cleaned Data
- Format: Simplified JSON with user_name, prompt_text, timestamp
- Contains: Already processed and filtered prompts
- Processing: Further filtering and session grouping

### Output Formats

#### Cleaned Data
```json
[
  {
    "user_name": "username",
    "prompt_text": "cleaned prompt text",
    "timestamp": "2023-01-01T00:00:00.000Z",
    "action_type": "fast"
  }
]
```

#### Reformatted Data (Session-based)
```json
[
  {
    "user_name": "username",
    "session_id": 1,
    "prompts": [
      {
        "prompt_text": "initial prompt",
        "timestamp": "2023-01-01T00:00:00.000Z",
        "changes": "Initial prompt"
      },
      {
        "prompt_text": "improved prompt",
        "timestamp": "2023-01-01T00:05:00.000Z",
        "changes": "Added: improved, better"
      }
    ]
  }
]
```

### Filtering Criteria

The script applies several filters to ensure high-quality training data:

1. **Meaningful Prompts**: Removes very short or low-quality prompts
2. **English Content**: Filters for English language prompts
3. **Duplicate Removal**: Eliminates case-insensitive duplicates
4. **Quality Checks**: Removes system commands, file paths, and parameter-only prompts
5. **Session Filtering**: For reformatted output, removes single-prompt sessions

### Examples

Show detailed usage examples:
```bash
python discord_data_cleaner.py --examples
```

### Integration with Training Data

The cleaned output from this script can be used to create training datasets for:

1. **Midjourney Prompt Refinement**: Using the session-based reformatted data
2. **Layout Prediction**: Using the cleaned prompt data
3. **General Text-to-Image Training**: Using the filtered prompt dataset

### Files in This Directory

- `discord_data_cleaner.py`: Main cleaning script
- `build_single_hop_jsonl.py`: Converts cleaned data to JSONL format for training
- `split_dataset.py`: Splits data into train/validation sets
- Various test and output files from processing runs

### Notes

- The script automatically detects and handles different Discord export formats
- Similarity threshold (0.2 default) can be adjusted based on your data characteristics
- Session grouping is useful for training models that understand prompt evolution
- All output files use UTF-8 encoding for international character support
