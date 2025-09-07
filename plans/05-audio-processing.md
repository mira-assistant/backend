# Audio Processing Architecture

## Overview

The audio processing system handles multi-client audio streams, quality assessment, and intelligent stream selection for the Mira network system. It's designed to work seamlessly across multiple devices while maintaining high audio quality and low latency.

## Core Components

### Multi-Stream Audio Collection
- **Concurrent Stream Handling**: Manages multiple audio streams simultaneously
- **Quality Assessment**: Real-time audio quality scoring
- **Stream Selection**: Intelligent selection of the best audio stream
- **Buffer Management**: Efficient audio buffer handling and processing

### Audio Quality Metrics
- **Signal-to-Noise Ratio (SNR)**: Measures signal quality relative to noise
- **Speech Clarity**: Analyzes speech intelligibility and clarity
- **Volume Level**: Monitors audio volume and consistency
- **Spectral Analysis**: Evaluates frequency content and characteristics

### Real-time Processing
- **Low Latency**: Minimal delay between audio input and processing
- **Streaming Processing**: Continuous audio stream processing
- **Adaptive Quality**: Dynamic quality adjustment based on conditions
- **Error Recovery**: Robust error handling and recovery mechanisms

## Audio Processing Pipeline

```
Audio Input → Preprocessing → Quality Analysis → Stream Selection → Processing
     ↓              ↓              ↓                ↓                ↓
Multiple Clients  Noise Reduction  Quality Metrics  Best Stream    Transcription
     ↓              ↓              ↓                ↓                ↓
Buffer Management  Filtering       Scoring          Selection        Speaker ID
     ↓              ↓              ↓                ↓                ↓
Queue Management   Normalization   Ranking          Processing       Context
```

## Multi-Stream Processor

### Stream Registration and Management
```python
class MultiStreamProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.active_streams = {}
        self.quality_scorer = AudioQualityScorer()
        self.stream_selector = StreamSelector()

    async def register_stream(self, network_id: str, client_id: str) -> bool:
        """Register a new audio stream"""
        stream_key = f"{network_id}:{client_id}"

        if stream_key in self.active_streams:
            return False

        self.active_streams[stream_key] = {
            "client_id": client_id,
            "network_id": network_id,
            "quality_metrics": QualityMetrics(),
            "last_update": datetime.now(),
            "buffer": AudioBuffer(),
            "is_active": True
        }

        return True

    async def process_audio_chunk(
        self,
        network_id: str,
        client_id: str,
        audio_data: bytes
    ) -> ProcessingResult:
        """Process a chunk of audio data"""
        stream_key = f"{network_id}:{client_id}"

        if stream_key not in self.active_streams:
            raise ValueError(f"Stream not registered: {stream_key}")

        stream = self.active_streams[stream_key]

        # Add audio data to buffer
        stream["buffer"].add_audio(audio_data)

        # Analyze quality if we have enough data
        if stream["buffer"].is_ready_for_analysis():
            quality_metrics = await self.quality_scorer.analyze(
                stream["buffer"].get_audio_data()
            )
            stream["quality_metrics"] = quality_metrics
            stream["last_update"] = datetime.now()

        # Select best stream for processing
        best_stream = await self.stream_selector.select_best_stream(
            network_id, self.active_streams
        )

        return ProcessingResult(
            client_id=client_id,
            quality_score=stream["quality_metrics"].overall_score,
            selected_for_processing=(best_stream == client_id),
            processing_priority=self.calculate_priority(stream)
        )
```

### Audio Quality Assessment
```python
class AudioQualityScorer:
    def __init__(self):
        self.weights = {
            "snr": 0.3,
            "speech_clarity": 0.3,
            "volume_level": 0.2,
            "consistency": 0.2
        }

    async def analyze(self, audio_data: np.ndarray) -> QualityMetrics:
        """Analyze audio quality and return comprehensive metrics"""

        # Calculate individual metrics
        snr = self.calculate_snr(audio_data)
        speech_clarity = self.calculate_speech_clarity(audio_data)
        volume_level = self.calculate_volume_level(audio_data)
        consistency = self.calculate_consistency(audio_data)

        # Calculate overall score
        overall_score = (
            self.weights["snr"] * snr +
            self.weights["speech_clarity"] * speech_clarity +
            self.weights["volume_level"] * volume_level +
            self.weights["consistency"] * consistency
        )

        return QualityMetrics(
            snr=snr,
            speech_clarity=speech_clarity,
            volume_level=volume_level,
            consistency=consistency,
            overall_score=overall_score,
            timestamp=datetime.now()
        )

    def calculate_snr(self, audio_data: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        if len(audio_data) == 0:
            return 0.0

        # Calculate signal power
        signal_power = np.var(audio_data)

        if signal_power == 0:
            return 0.0

        # Estimate noise power using quiet segments
        window_size = len(audio_data) // 10
        if window_size < 100:
            noise_power = signal_power * 0.1
        else:
            windowed_power = []
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                windowed_power.append(np.var(window))

            # Use bottom 20% as noise estimate
            windowed_power.sort()
            noise_power = np.mean(windowed_power[:max(1, len(windowed_power) // 5)])

        if noise_power <= 0:
            noise_power = signal_power * 0.01

        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(0.0, min(100.0, snr_db))

    def calculate_speech_clarity(self, audio_data: np.ndarray) -> float:
        """Calculate speech clarity based on spectral analysis"""
        if len(audio_data) == 0:
            return 0.0

        # Calculate power spectral density
        freqs, psd = signal.welch(audio_data, fs=16000, nperseg=min(1024, len(audio_data) // 4))

        # Speech frequency range (300-3400 Hz)
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        speech_power = np.sum(psd[speech_mask])
        total_power = np.sum(psd)

        if total_power == 0:
            return 0.0

        # Speech clarity as ratio of speech-band power
        speech_ratio = speech_power / total_power

        # Calculate spectral flatness for clarity
        if np.any(psd[speech_mask] > 0):
            geometric_mean = np.exp(np.mean(np.log(psd[speech_mask] + 1e-10)))
            arithmetic_mean = np.mean(psd[speech_mask])
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            clarity_factor = 1.0 - spectral_flatness
        else:
            clarity_factor = 0.0

        # Combine metrics
        clarity_score = (speech_ratio * 0.7 + clarity_factor * 0.3) * 100
        return min(100.0, max(0.0, clarity_score))
```

### Stream Selection Algorithm
```python
class StreamSelector:
    def __init__(self):
        self.selection_history = {}
        self.quality_threshold = 0.6
        self.stability_window = 5  # seconds

    async def select_best_stream(
        self,
        network_id: str,
        active_streams: dict
    ) -> str:
        """Select the best audio stream for processing"""

        # Filter streams for this network
        network_streams = {
            k: v for k, v in active_streams.items()
            if v["network_id"] == network_id and v["is_active"]
        }

        if not network_streams:
            return None

        # Calculate selection scores
        scores = {}
        for stream_key, stream in network_streams.items():
            score = self.calculate_selection_score(stream)
            scores[stream_key] = score

        # Select stream with highest score
        best_stream_key = max(scores, key=scores.get)
        best_stream = network_streams[best_stream_key]

        # Update selection history
        self.update_selection_history(network_id, best_stream["client_id"])

        return best_stream["client_id"]

    def calculate_selection_score(self, stream: dict) -> float:
        """Calculate selection score for a stream"""
        metrics = stream["quality_metrics"]

        # Base score from quality metrics
        base_score = metrics.overall_score

        # Stability bonus (prefer consistent streams)
        stability_bonus = self.calculate_stability_bonus(stream)

        # Recency bonus (prefer recently updated streams)
        recency_bonus = self.calculate_recency_bonus(stream)

        # Combine scores
        total_score = (
            base_score * 0.7 +
            stability_bonus * 0.2 +
            recency_bonus * 0.1
        )

        return min(100.0, max(0.0, total_score))

    def calculate_stability_bonus(self, stream: dict) -> float:
        """Calculate stability bonus based on quality consistency"""
        # This would track quality over time and reward consistency
        # For now, return a base value
        return 50.0

    def calculate_recency_bonus(self, stream: dict) -> float:
        """Calculate recency bonus for recently updated streams"""
        time_since_update = (datetime.now() - stream["last_update"]).total_seconds()

        if time_since_update < 1.0:  # Less than 1 second
            return 100.0
        elif time_since_update < 5.0:  # Less than 5 seconds
            return 80.0
        else:
            return max(0.0, 100.0 - time_since_update * 10)
```

## Audio Preprocessing

### Noise Reduction
```python
class AudioPreprocessor:
    def __init__(self):
        self.denoiser = NoiseReducer()
        self.filter = AudioFilter()
        self.normalizer = AudioNormalizer()

    async def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio data for optimal quality"""

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize audio
        audio_data = self.normalizer.normalize(audio_data)

        # Apply noise reduction
        audio_data = await self.denoiser.reduce_noise(audio_data)

        # Apply filtering
        audio_data = self.filter.apply_filters(audio_data)

        return audio_data

class NoiseReducer:
    def __init__(self):
        self.noise_profile = None
        self.adaptation_rate = 0.1

    async def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce noise in audio data"""
        try:
            # Use first part of audio as noise sample
            noise_sample = audio_data[:len(audio_data) // 4]

            # Apply noise reduction
            denoised = nr.reduce_noise(
                y=audio_data,
                sr=16000,
                y_noise=noise_sample,
                prop_decrease=0.8,
                stationary=False
            )

            return denoised.astype(np.float32)

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data

class AudioFilter:
    def __init__(self):
        self.high_pass_cutoff = 80  # Hz
        self.low_pass_cutoff = 8000  # Hz

    def apply_filters(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply high-pass and low-pass filters"""
        # High-pass filter to remove low-frequency noise
        audio_data = self.high_pass_filter(audio_data, self.high_pass_cutoff)

        # Low-pass filter to remove high-frequency noise
        audio_data = self.low_pass_filter(audio_data, self.low_pass_cutoff)

        return audio_data

    def high_pass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply high-pass Butterworth filter"""
        nyquist = 0.5 * 16000
        normal_cutoff = cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    def low_pass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply low-pass Butterworth filter"""
        nyquist = 0.5 * 16000
        normal_cutoff = cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)
```

## Real-time Audio Buffer Management

### Circular Buffer Implementation
```python
class AudioBuffer:
    def __init__(self, max_size: int = 16000 * 10):  # 10 seconds at 16kHz
        self.buffer = np.zeros(max_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.max_size = max_size
        self.is_full = False

    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to buffer"""
        data_len = len(audio_data)

        if data_len > self.max_size:
            # If data is larger than buffer, take the last part
            audio_data = audio_data[-self.max_size:]
            data_len = len(audio_data)

        # Write data to buffer
        if self.write_pos + data_len <= self.max_size:
            # Simple case: data fits without wrapping
            self.buffer[self.write_pos:self.write_pos + data_len] = audio_data
            self.write_pos += data_len
        else:
            # Wrap around case
            first_part = self.max_size - self.write_pos
            second_part = data_len - first_part

            self.buffer[self.write_pos:] = audio_data[:first_part]
            self.buffer[:second_part] = audio_data[first_part:]
            self.write_pos = second_part

        # Update read position if buffer is full
        if self.write_pos == self.read_pos and not self.is_full:
            self.is_full = True
            self.read_pos = (self.read_pos + data_len) % self.max_size

    def get_audio_data(self, duration_seconds: float = 3.0) -> np.ndarray:
        """Get audio data from buffer"""
        samples_needed = int(16000 * duration_seconds)

        if self.is_full:
            # Buffer is full, get most recent data
            if samples_needed >= self.max_size:
                return self.buffer.copy()
            else:
                start_pos = (self.write_pos - samples_needed) % self.max_size
                if start_pos + samples_needed <= self.max_size:
                    return self.buffer[start_pos:start_pos + samples_needed]
                else:
                    first_part = self.max_size - start_pos
                    second_part = samples_needed - first_part
                    return np.concatenate([
                        self.buffer[start_pos:],
                        self.buffer[:second_part]
                    ])
        else:
            # Buffer not full, get available data
            available_samples = self.write_pos - self.read_pos
            samples_to_get = min(samples_needed, available_samples)
            return self.buffer[self.read_pos:self.read_pos + samples_to_get]

    def is_ready_for_analysis(self) -> bool:
        """Check if buffer has enough data for analysis"""
        if self.is_full:
            return True

        available_samples = self.write_pos - self.read_pos
        return available_samples >= 16000 * 2  # 2 seconds minimum
```

## Performance Optimization

### Async Processing
```python
class AsyncAudioProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.processing_queue = asyncio.Queue()
        self.workers = []

    async def start_workers(self):
        """Start background processing workers"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.audio_worker(f"worker_{i}"))
            self.workers.append(worker)

    async def audio_worker(self, worker_id: str):
        """Background worker for audio processing"""
        while True:
            try:
                # Get audio processing task
                task = await self.processing_queue.get()

                # Process audio
                result = await self.process_audio_task(task)

                # Send result
                await self.send_result(result)

                # Mark task as done
                self.processing_queue.task_done()

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    async def queue_audio_processing(
        self,
        network_id: str,
        client_id: str,
        audio_data: bytes
    ):
        """Queue audio for processing"""
        task = AudioProcessingTask(
            network_id=network_id,
            client_id=client_id,
            audio_data=audio_data,
            timestamp=datetime.now()
        )

        await self.processing_queue.put(task)
```

### Caching and Optimization
```python
class AudioProcessingCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def get_cached_result(self, audio_hash: str) -> Optional[ProcessingResult]:
        """Get cached processing result"""
        if audio_hash in self.cache:
            self.access_times[audio_hash] = datetime.now()
            return self.cache[audio_hash]
        return None

    def cache_result(self, audio_hash: str, result: ProcessingResult):
        """Cache processing result"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]

        self.cache[audio_hash] = result
        self.access_times[audio_hash] = datetime.now()

    def calculate_audio_hash(self, audio_data: bytes) -> str:
        """Calculate hash for audio data"""
        return hashlib.md5(audio_data).hexdigest()
```

## Error Handling and Recovery

### Error Classification
```python
class AudioProcessingError(Exception):
    def __init__(self, error_type: str, message: str, recoverable: bool = True):
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)

class AudioQualityError(AudioProcessingError):
    def __init__(self, message: str):
        super().__init__("audio_quality", message, True)

class StreamConnectionError(AudioProcessingError):
    def __init__(self, message: str):
        super().__init__("stream_connection", message, False)
```

### Recovery Mechanisms
```python
class AudioRecoveryManager:
    def __init__(self):
        self.retry_attempts = {}
        self.max_retries = 3

    async def handle_processing_error(
        self,
        error: AudioProcessingError,
        network_id: str,
        client_id: str
    ):
        """Handle audio processing errors with recovery"""

        if not error.recoverable:
            await self.notify_unrecoverable_error(error, network_id, client_id)
            return

        # Check retry attempts
        error_key = f"{network_id}:{client_id}"
        attempts = self.retry_attempts.get(error_key, 0)

        if attempts >= self.max_retries:
            await self.notify_max_retries_exceeded(error, network_id, client_id)
            return

        # Increment retry count
        self.retry_attempts[error_key] = attempts + 1

        # Wait before retry (exponential backoff)
        wait_time = 2 ** attempts
        await asyncio.sleep(wait_time)

        # Attempt recovery
        await self.attempt_recovery(network_id, client_id)

    async def attempt_recovery(self, network_id: str, client_id: str):
        """Attempt to recover from error"""
        try:
            # Reset stream state
            await self.reset_stream_state(network_id, client_id)

            # Notify client to resend audio
            await self.request_audio_resend(network_id, client_id)

        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
```

This audio processing architecture provides a robust, scalable foundation for handling multi-client audio streams in the Mira network system while maintaining high quality and performance.



