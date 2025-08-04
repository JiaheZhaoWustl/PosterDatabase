import json
import random
import os

def split_jsonl_dataset(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    """
    Split a JSONL dataset into training and validation sets.
    
    Args:
        input_file (str): Path to the input JSONL file
        train_file (str): Path to the output training file
        val_file (str): Path to the output validation file
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        seed (int): Random seed for reproducibility (default: 42)
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Shuffle the lines
    random.shuffle(lines)
    
    # Calculate split point
    split_point = int(len(lines) * train_ratio)
    
    # Split the data
    train_lines = lines[:split_point]
    val_lines = lines[split_point:]
    
    # Write training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    # Write validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    print(f"Dataset split completed:")
    print(f"Total samples: {len(lines)}")
    print(f"Training samples: {len(train_lines)} ({len(train_lines)/len(lines)*100:.1f}%)")
    print(f"Validation samples: {len(val_lines)} ({len(val_lines)/len(lines)*100:.1f}%)")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")

if __name__ == "__main__":
    # File paths
    input_file = "newbies-41_single_hop.jsonl"
    train_file = "newbies-41_single_hop_train.jsonl"
    val_file = "newbies-41_single_hop_val.jsonl"
    
    # Split the dataset
    split_jsonl_dataset(input_file, train_file, val_file) 