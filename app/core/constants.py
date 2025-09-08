"""
Application constants that don't change between environments.
These are business logic constants, not configuration settings.
"""

# Audio Processing Constants
SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
AUDIO_FORMAT = "wav"

# Context Processing Constants
CONVERSATION_GAP_THRESHOLD = 300  # seconds
CONTEXT_SIMILARITY_THRESHOLD = 0.7
MAX_CONTEXT_LENGTH = 4000  # tokens

# API Constants
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Application Constants
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Model Constants
DEFAULT_MODEL_TEMPERATURE = 0.7
MAX_TOKENS = 2000

# File Processing Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_AUDIO_FORMATS = ["wav", "mp3", "m4a", "flac"]

# Database Constants
DEFAULT_POOL_SIZE = 5
MAX_POOL_SIZE = 20
