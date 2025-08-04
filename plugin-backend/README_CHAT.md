# Chat Bot Feature for Plugin Backend

This document describes the new chat bot functionality that allows users to converse with your fine-tuned 4o-mini model for poster design assistance.

## Overview

The chat bot feature provides:
- **Conversational AI**: Users can chat with your fine-tuned model for design guidance
- **Context Awareness**: Maintains conversation history for coherent multi-turn discussions
- **Error Handling**: Robust retry logic and graceful error handling
- **Easy Integration**: Simple API endpoints for frontend integration

## Files Added

### Core Files
- `chat_bot.py` - Main chat bot functionality module
- `server.py` - Updated with new chat endpoints
- `test_chat.py` - Test script for the chat bot functionality
- `chat_ui.html` - Simple web interface for testing

## API Endpoints

### POST `/chat`
Send a message to the AI assistant.

**Request Body:**
```json
{
  "message": "Hello! I need help with my poster design.",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ],
  "temperature": 0.7,
  "model": "ft:gpt-4o-mini-2024-07-18:sia-project-1:jun24-test:Bm0QDLkZ"
}
```

**Response:**
```json
{
  "response": "I'd be happy to help with your poster design!",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"},
    {"role": "user", "content": "Hello! I need help with my poster design."},
    {"role": "assistant", "content": "I'd be happy to help with your poster design!"}
  ],
  "model_used": "ft:gpt-4o-mini-2024-07-18:sia-project-1:jun24-test:Bm0QDLkZ"
}
```

### POST `/chat/reset`
Reset the conversation history.

**Response:**
```json
{
  "message": "Conversation history reset",
  "conversation_history": []
}
```

## Usage Examples

### Python Script
```python
from chat_bot import chat_with_model

# Start a new conversation
result = chat_with_model(
    user_message="Hello! I'm working on a poster design.",
    conversation_history=[],
    model="ft:gpt-4o-mini-2024-07-18:sia-project-1:jun24-test:Bm0QDLkZ"
)

print(result['response'])

# Continue the conversation
result = chat_with_model(
    user_message="What about the color scheme?",
    conversation_history=result['conversation_history']
)

print(result['response'])
```

### Command Line Testing
```bash
# Test with a single message
python chat_bot.py --message "Hello! I need help with my poster design."

# Test with conversation history
python chat_bot.py --message "What about the color scheme?" --history conversation.json

# Test with different model
python chat_bot.py --message "Hello!" --model "your-model-id" --temp 0.8
```

### Web Interface
1. Start the Flask server: `python server.py`
2. Open `chat_ui.html` in your browser
3. Start chatting with the AI assistant

## Configuration

### Model Settings
Update the `DEFAULT_CHAT_MODEL` in `chat_bot.py` to use your specific fine-tuned model:

```python
DEFAULT_CHAT_MODEL = "ft:your-model-id-here"
```

### System Message
Customize the system message in `chat_bot.py` to match your use case:

```python
system_msg = """You are a helpful AI assistant for poster design and layout. 
Help users with their poster design questions, provide suggestions, and guide them through the design process.
Be conversational, friendly, and provide practical advice."""
```

### Temperature and Retry Settings
Adjust these parameters in the `chat_with_model` function:
- `temp`: Controls response creativity (0.0-1.0)
- `max_retries`: Number of retry attempts on API errors
- `retry_wait`: Wait time between retries

## Testing

### Run the Test Script
```bash
python test_chat.py
```

This will test:
- Basic chat bot functionality
- Conversation with history
- Chat reset functionality

### Manual Testing with Web Interface
1. Start the server: `python server.py`
2. Open `chat_ui.html` in your browser
3. Test the conversation flow

## Integration with Frontend

### JavaScript Example
```javascript
async function sendMessage(message, conversationHistory = []) {
    const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            conversation_history: conversationHistory,
            temperature: 0.7
        })
    });
    
    const data = await response.json();
    return data;
}

// Usage
const result = await sendMessage("Hello!", []);
console.log(result.response);
```

## Error Handling

The chat bot functionality includes comprehensive error handling:

- **API Timeouts**: Automatic retry with exponential backoff
- **Rate Limits**: Graceful handling with retry logic
- **Network Errors**: User-friendly error messages
- **Invalid Input**: Validation and helpful error responses

## Security Considerations

- Ensure your API key is properly secured in environment variables
- Consider rate limiting for production use
- Validate user input before sending to the model
- Monitor API usage and costs

## Next Steps

1. **Test the chat bot functionality** with your specific fine-tuned model
2. **Customize the system message** to match your use case
3. **Integrate with your frontend** using the provided API endpoints
4. **Add additional features** like conversation export, user preferences, etc.
5. **Deploy to production** with proper security measures

## Troubleshooting

### Common Issues

1. **Connection Error**: Make sure the Flask server is running on port 5000
2. **API Key Error**: Ensure `OPENAI_API_KEY` is set in your environment
3. **Model Not Found**: Verify your fine-tuned model ID is correct
4. **CORS Issues**: The server includes CORS headers, but check browser console for issues

### Debug Mode
Enable debug logging by modifying the logging level in `chat_bot.py`:

```python
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
```

## Support

For issues or questions about the chat bot functionality, check:
1. The test script output for error details
2. Flask server logs for API errors
3. Browser console for frontend issues
4. OpenAI API documentation for model-specific issues 