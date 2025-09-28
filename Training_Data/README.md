# Training Data

This directory contains training datasets for two fine-tuned models, designed to improve model performance, on ChatGPT's platform:
1. predicting typographic layout of event posters
2. midjourney's text-to-image prompt refinement

## Dataset Overview

### Midjourney Chat Datasets
- **`midjourney_chat_train.jsonl`** (1.5MB) - Training data for chat-based poster generation
- **`midjourney_chat_validation.jsonl`** (376KB) - Validation data for chat-based poster generation  
- **`midjourney_chat_full_data.jsonl`** (1.8MB) - Complete dataset combining training and validation

### Layout Prediction Datasets
- **`layout_prediction_train.jsonl`** (1.1MB) - Training data for layout prediction tasks
- **`layout_prediction_validation.jsonl`** (123KB) - Validation data for layout prediction tasks

## Usage for ChatGPT Platform Training

### Prerequisites
- Access to ChatGPT's fine-tuning platform
- API credentials for model training
- Understanding of JSONL format for training data

### Training Process

1. **Prepare Your Data**: Ensure your training data follows the JSONL format with appropriate conversation structures
2. **Upload to Platform**: Use ChatGPT's platform to upload your training files
3. **Configure Training**: Set appropriate hyperparameters based on your dataset size and complexity
4. **Monitor Training**: Track training progress and validation metrics
5. **Evaluate Results**: Test the fine-tuned model on your specific use cases

### Data Format
Each line in the JSONL files contains a JSON object with conversation data structured for training. The exact format depends on the specific task (chat-based generation vs. layout prediction).

## Open Source Training

If you want to improve model performance with your own data:

1. **Collect Domain-Specific Data**: Gather examples relevant to your use case
2. **Format Consistently**: Structure your data to match the existing format
3. **Maintain Quality**: Ensure high-quality, diverse training examples
4. **Split Appropriately**: Create separate training and validation sets
5. **Iterate**: Continuously improve your dataset based on model performance

## Dataset Statistics

- **Total Training Examples**: ~1,500+ across all datasets
- **Validation Split**: ~20% of total data
- **Data Sources**: Curated from real-world poster generation scenarios
- **Format**: JSONL (JSON Lines) for platform compatibility

## Contributing

To contribute improved training data:

1. Follow the existing data format and structure
2. Ensure data quality and relevance
3. Include appropriate validation splits
4. Document any new data sources or preprocessing steps
5. Test the data with existing training pipelines

## Notes

- These datasets are specifically formatted for ChatGPT's fine-tuning platform
- The JSONL format ensures compatibility with the training pipeline
- Validation sets are crucial for monitoring training progress and preventing overfitting
- Consider the balance between training and validation data when creating new datasets

For questions about the data format or training process, refer to ChatGPT's fine-tuning documentation or the main project README.
