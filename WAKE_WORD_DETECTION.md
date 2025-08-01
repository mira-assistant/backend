# Wake Word Detection System

The Wake Word Detection System provides real-time monitoring and detection of trigger words and phrases across multiple audio streams. It integrates seamlessly with the existing audio stream scoring system to enable voice-activated functionality.

## Features

### Core Functionality
- **Real-time wake word detection** across multiple audio streams
- **Configurable wake words** with custom sensitivity settings
- **Case-insensitive detection** with fuzzy matching capabilities
- **Cooldown periods** to prevent rapid re-triggering
- **Multi-client support** with individual tracking per client
- **Thread-safe operation** for concurrent audio processing

### Advanced Features
- **Detection callbacks** for custom action triggering
- **Recent detection history** with configurable limits
- **Statistics and monitoring** for system performance analysis
- **Enable/disable controls** for individual wake words and entire system
- **Placeholder for ML-based detection** when advanced dependencies are available

## Default Wake Words

The system comes pre-configured with the following wake words:
- `hey mira`
- `okay mira`
- `mira`
- `listen mira`
- `start recording`

## API Endpoints

### Wake Word Management

#### Get All Wake Words
```http
GET /wake-words
```

**Response:**
```json
{
  "wake_words": {
    "hey mira": {
      "word": "hey mira",
      "sensitivity": 0.7,
      "enabled": true,
      "min_confidence": 0.5,
      "cooldown_seconds": 2.0
    }
  },
  "stats": {
    "enabled": true,
    "total_wake_words": 5,
    "enabled_wake_words": 5,
    "total_detections": 12,
    "detections_by_wake_word": {
      "hey mira": 8,
      "mira": 4
    },
    "active_callbacks": 0,
    "numpy_available": false
  }
}
```

#### Add Wake Word
```http
POST /wake-words
Content-Type: application/json

{
  "word": "hello assistant",
  "sensitivity": 0.8,
  "min_confidence": 0.6,
  "cooldown_seconds": 3.0
}
```

**Response:**
```json
{
  "message": "Wake word 'hello assistant' added successfully",
  "word": "hello assistant",
  "sensitivity": 0.8,
  "min_confidence": 0.6,
  "cooldown_seconds": 3.0
}
```

#### Remove Wake Word
```http
DELETE /wake-words/{word}
```

**Response:**
```json
{
  "message": "Wake word 'hello assistant' removed successfully",
  "word": "hello assistant"
}
```

#### Enable/Disable Wake Word
```http
PATCH /wake-words/{word}/enable
PATCH /wake-words/{word}/disable
```

**Response:**
```json
{
  "message": "Wake word 'hello assistant' enabled",
  "word": "hello assistant",
  "enabled": true
}
```

### System Control

#### Enable/Disable Wake Word Detection System
```http
PATCH /wake-words/enable
PATCH /wake-words/disable
```

**Response:**
```json
{
  "message": "Wake word detection enabled",
  "enabled": true
}
```

### Detection Management

#### Get Recent Detections
```http
GET /wake-words/detections?limit=10
```

**Response:**
```json
{
  "detections": [
    {
      "wake_word": "hey mira",
      "confidence": 1.0,
      "client_id": "phone_client",
      "timestamp": "2023-12-07T10:30:45.123456Z",
      "audio_snippet_length": 2.5
    }
  ],
  "count": 1,
  "limit": 10
}
```

#### Clear All Detections
```http
DELETE /wake-words/detections
```

**Response:**
```json
{
  "message": "All wake word detections cleared",
  "cleared": true
}
```

#### Process Text for Wake Words
```http
POST /wake-words/process
Content-Type: application/json

{
  "client_id": "test_client",
  "text": "Hey Mira, what's the weather today?",
  "audio_length": 2.5
}
```

**Response (Detection Found):**
```json
{
  "detected": true,
  "wake_word": "hey mira",
  "confidence": 1.0,
  "client_id": "test_client",
  "timestamp": "2023-12-07T10:30:45.123456Z",
  "audio_snippet_length": 2.5
}
```

**Response (No Detection):**
```json
{
  "detected": false,
  "message": "No wake word detected in the provided text"
}
```

## Integration with Audio Stream Scoring

The wake word detection system automatically integrates with the audio processing pipeline:

1. **Automatic Detection**: When audio is processed through `/interactions/register`, the transcribed text is automatically checked for wake words
2. **Client Tracking**: Wake word detections are associated with specific clients from the stream scoring system
3. **Audio Length Calculation**: The system calculates audio snippet length from the raw audio data
4. **Logging**: All wake word detections are logged with appropriate detail levels

### Example Integration Flow

```python
# In the audio processing pipeline (mira.py)
if transcription_result and transcription_result.get("text"):
    # Calculate audio length from bytes
    audio_length = len(sentence_buf_raw) / (16000 * 2)
    
    # Check for wake words
    wake_word_detection = wake_word_detector.process_audio_text(
        client_id=client_id or "unknown",
        transcribed_text=transcription_result["text"],
        audio_length=audio_length
    )
    
    if wake_word_detection:
        logger.info(f"Wake word '{wake_word_detection.wake_word}' detected")
        # Trigger custom actions here
```

## Configuration Parameters

### Wake Word Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | string | - | The wake word or phrase to detect |
| `sensitivity` | float | 0.7 | Detection threshold (0.0-1.0) |
| `enabled` | boolean | true | Whether this wake word is enabled |
| `min_confidence` | float | 0.5 | Minimum confidence required for detection |
| `cooldown_seconds` | float | 2.0 | Time to wait before detecting the same word again |

### Detection Algorithm

The wake word detection uses a multi-level approach:

1. **Exact Match**: Perfect string matching gets confidence = 1.0
2. **Partial Match**: Word-by-word matching for multi-word phrases
3. **Fuzzy Match**: Character similarity for single words
4. **Threshold Check**: Only detections above `min_confidence` are reported

### Confidence Scoring

- **Exact match**: 1.0
- **Substring match**: 0.8
- **Character similarity**: Based on common characters (minimum 0.6)
- **Multi-word partial**: Ratio of matching words

## Example Usage

### Basic Wake Word Setup

```python
from wake_word_detector import WakeWordDetector

# Initialize detector
detector = WakeWordDetector(sample_rate=16000)

# Add custom wake word
detector.add_wake_word("hello assistant", sensitivity=0.8)

# Process audio text
detection = detector.process_audio_text(
    client_id="phone_client",
    transcribed_text="Hello assistant, turn on the lights",
    audio_length=2.5
)

if detection:
    print(f"Detected: {detection.wake_word} (confidence: {detection.confidence})")
```

### Using Detection Callbacks

```python
def on_wake_word_detected(detection):
    print(f"Wake word '{detection.wake_word}' detected from {detection.client_id}")
    # Trigger custom actions:
    # - Start recording
    # - Change system state
    # - Send notifications
    # - Execute commands

# Add callback
detector.add_detection_callback(on_wake_word_detected)
```

### Configuration Management

```python
# Get all wake words
wake_words = detector.get_wake_words()

# Enable/disable specific wake words
detector.set_wake_word_enabled("hey mira", False)

# Enable/disable entire system
detector.set_enabled(False)

# Get statistics
stats = detector.get_stats()
print(f"Total detections: {stats['total_detections']}")
```

## Future Enhancements

### Planned Features

1. **ML-based Audio Detection**: Direct audio processing using machine learning models when scipy/numpy dependencies are available
2. **Custom Model Support**: Integration with pre-trained wake word detection models
3. **Voice Print Matching**: Speaker-specific wake word detection
4. **Adaptive Sensitivity**: Automatic sensitivity adjustment based on detection accuracy
5. **Audio Preprocessing**: Noise reduction and audio enhancement for better detection

### ML Integration Placeholder

The system includes placeholder infrastructure for advanced ML-based detection:

```python
def process_audio_raw(self, client_id: str, audio_data, audio_length: float = 0.0):
    """
    Process raw audio data for wake word detection.
    
    This is a placeholder for advanced audio-based wake word detection.
    Future implementations could include:
    - Audio preprocessing (noise reduction, normalization)
    - Feature extraction (MFCC, spectrograms)
    - ML model inference for wake word detection
    - Confidence scoring and thresholding
    """
    # Implementation coming when ML dependencies are available
    return None
```

## Testing

The wake word detection system includes comprehensive test coverage:

- **46 unit tests** covering core functionality and edge cases
- **19 API integration tests** validating all endpoints
- **Error handling tests** for invalid inputs and edge cases
- **Concurrent operation tests** for thread safety
- **Callback system tests** for custom integration

### Running Tests

```bash
# Test the wake word detector core functionality
pytest test_wake_word_detector.py -v

# Test the API endpoints
pytest test_wake_word_api.py -v

# Run all tests
pytest --tb=short
```

## Performance Considerations

### Efficiency
- **Text-based detection** is lightweight and fast
- **Thread-safe operations** with minimal locking
- **Configurable cooldowns** prevent excessive processing
- **Limited history storage** (default: 50 recent detections)

### Memory Usage
- **Minimal memory footprint** for text processing
- **Automatic cleanup** of old detections
- **No heavy ML dependencies** required for basic functionality

### Scalability
- **Multiple client support** with individual tracking
- **Concurrent audio processing** across streams
- **Configurable detection limits** and thresholds
- **Extensible callback system** for custom integrations

This wake word detection system provides a robust foundation for voice-activated functionality while maintaining compatibility with existing audio processing pipelines and preparing for future ML-based enhancements.