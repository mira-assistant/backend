# Audio Stream Scoring System

This document describes the audio stream scoring system implemented in the Mira Assistant backend for dynamically selecting the best audio stream from connected listening clients.

## Overview

The audio stream scoring system evaluates multiple concurrent audio streams based on objective quality metrics and automatically selects the best performing stream for optimal recording quality. The system is designed for real-time operation and can handle multiple clients simultaneously.

## Key Features

### 1. Signal Quality Metrics
- **Signal-to-Noise Ratio (SNR)**: Measures the ratio of signal power to noise power in dB
- **Speech Clarity**: Analyzes spectral characteristics in the speech frequency band (300-3400 Hz) 
- **Volume Level**: RMS volume measurement for stream activity detection
- **Noise Level**: Estimated background noise level

### 2. Real-time Stream Selection
- Continuously evaluates all active streams
- Automatically switches to the best performing stream
- Maintains history of quality scores for trend analysis

### 3. Future-Ready Design
- **Phone Distance Placeholder**: Built-in support for incorporating proximity data from mobile devices
- **Extensible Scoring Weights**: Configurable weights for different quality metrics
- **Device Type Awareness**: Support for different device characteristics

## Architecture

### Core Components

#### AudioStreamScorer Class
The main scoring engine that handles:
- Client registration and management
- Real-time quality metric calculation
- Best stream selection logic
- Cleanup of inactive clients

#### API Endpoints
New endpoints added to the FastAPI application:
- `GET /streams/best` - Get current best stream
- `GET /streams/scores` - Get scores for all active streams  
- `POST /streams/{client_id}/distance` - Set phone distance (future feature)
- `GET /streams/{client_id}/info` - Get detailed client stream info
- `POST /streams/cleanup` - Remove inactive streams

#### Integration Points
- Enhanced client registration with metadata support
- Integration with existing audio processing pipeline
- Updated interaction registration to include stream quality updates

## Usage

### Client Registration
```python
# Register a client with metadata
POST /service/client/register/my_client?device_type=phone
{
    "location": {"lat": 37.7749, "lng": -122.4194}
}
```

### Getting Best Stream
```python
# Get the currently selected best stream
GET /streams/best
```

Response:
```json
{
    "best_stream": {
        "client_id": "my_client",
        "score": 85.2,
        "metrics": {
            "snr": 22.5,
            "speech_clarity": 78.3,
            "volume_level": 0.1245,
            "noise_level": 0.0156,
            "phone_distance": null,
            "last_update": "2023-12-01T10:30:00Z",
            "sample_count": 15
        }
    }
}
```

### Setting Phone Distance (Future Feature)
```python
# Set distance for proximity-based scoring
POST /streams/my_client/distance
{
    "distance": 2.5
}
```

## Scoring Algorithm

The overall stream score is calculated using weighted metrics:

```
Overall Score = (SNR_score × 0.4) + (Clarity_score × 0.4) + (Volume_score × 0.1) + (Distance_score × 0.1)
```

### Metric Calculations

**SNR Calculation:**
- Estimates signal power using audio variance
- Estimates noise power from quieter portions (bottom 20% of windowed power)
- Returns SNR in dB with non-negative values

**Speech Clarity:**
- Analyzes power spectral density in speech frequency range (300-3400 Hz)
- Considers spectral flatness to measure speech-like characteristics
- Returns score from 0-100

**Volume Level:**
- RMS (Root Mean Square) calculation of audio amplitude
- Used to detect stream activity and relative loudness

**Distance Score (Placeholder):**
- Currently returns 100 if no distance data available
- Future implementation will favor closer devices

## Real-time Operation

The system is designed for minimal latency:
- Quality metrics updated on each audio interaction
- Best stream selection computed immediately after updates
- Thread-safe operations using locks for concurrent access
- Automatic cleanup of inactive clients

## Configuration

Scoring weights can be adjusted in the AudioStreamScorer initialization:

```python
self.weights = {
    'snr': 0.4,           # Signal-to-noise ratio importance
    'speech_clarity': 0.4, # Speech clarity importance
    'volume_level': 0.1,   # Volume level importance  
    'phone_distance': 0.1  # Distance importance (future)
}
```

## Testing

The system includes comprehensive tests:
- Unit tests for the AudioStreamScorer class (`test_audio_stream_scorer.py`)
- API endpoint tests (`test_api_minimal.py`)
- Integration tests with the main application

Run tests with:
```bash
pytest test_audio_stream_scorer.py -v
python test_api_minimal.py
```

## Future Enhancements

### Phone Distance Integration
The system is designed to easily incorporate phone distance data when the mobile app is ready:

1. **Location Services**: GPS/Bluetooth beacon distance calculation
2. **Proximity Scoring**: Closer devices get higher scores
3. **Multi-device Optimization**: Coordinate multiple phone streams

### Advanced Features
- **Adaptive Weights**: Machine learning-based weight optimization
- **Stream Quality Prediction**: Predictive quality scoring
- **Environmental Adaptation**: Adjust scoring based on acoustic environment
- **Stream Mixing**: Combine multiple high-quality streams

## Error Handling

The system includes robust error handling:
- Graceful degradation when audio analysis fails
- Automatic client cleanup for disconnected streams
- Fallback scoring when metrics cannot be calculated
- Comprehensive logging for debugging

## Performance Considerations

- **Memory Efficient**: Limited history storage (last 10 scores per client)
- **CPU Optimized**: Efficient spectral analysis with adaptive window sizing
- **Thread Safe**: Concurrent operations supported
- **Scalable**: Designed to handle dozens of concurrent streams

## Integration with Existing System

The scoring system integrates seamlessly with the existing Mira backend:
- **Non-intrusive**: Existing functionality remains unchanged
- **Optional**: Stream scoring can be disabled without affecting core features
- **Compatible**: Works with existing client registration and audio processing
- **Future-proof**: Designed for easy integration with upcoming mobile app