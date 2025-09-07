# Migration Plan: Legacy to Network-Based Architecture

## Overview

This document outlines the step-by-step migration plan from the current legacy single-user system to the new multi-network, multi-client architecture. The migration is designed to be gradual, with minimal downtime and backward compatibility.

## Migration Strategy

### Phased Approach
1. **Phase 1**: Database schema migration and network support
2. **Phase 2**: API layer migration and client registration
3. **Phase 3**: Audio processing and multi-client support
4. **Phase 4**: WebSocket implementation and real-time sync
5. **Phase 5**: Legacy system deprecation and cleanup

### Backward Compatibility
- **Legacy API Support**: Maintain legacy endpoints during transition
- **Data Migration**: Preserve all existing data and relationships
- **Gradual Rollout**: Migrate networks one at a time
- **Rollback Plan**: Ability to revert to legacy system if needed

## Phase 1: Database Schema Migration

### Step 1.1: Create New Database Schema
```sql
-- Create new tables for network-based architecture
-- (See 02-database-design.md for complete schema)

-- Create networks table
CREATE TABLE networks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    -- ... other fields
);

-- Create clients table
CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id),
    -- ... other fields
);

-- Add network_id columns to existing tables
ALTER TABLE persons ADD COLUMN network_id UUID;
ALTER TABLE interactions ADD COLUMN network_id UUID;
ALTER TABLE conversations ADD COLUMN network_id UUID;
ALTER TABLE actions ADD COLUMN network_id UUID;
```

### Step 1.2: Migrate Existing Data
```python
# Migration script: migrate_legacy_data.py
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def migrate_legacy_data():
    """Migrate existing data to network-based schema"""

    # Create default network for legacy data
    legacy_network_id = "legacy_migration_network"

    # Create network record
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO networks (network_id, name, status, created_at)
            VALUES (:network_id, :name, 'active', NOW())
        """), {
            "network_id": legacy_network_id,
            "name": "Legacy Data Migration"
        })

        # Get the network UUID
        result = conn.execute(text("""
            SELECT id FROM networks WHERE network_id = :network_id
        """), {"network_id": legacy_network_id})
        network_uuid = result.fetchone()[0]

        # Update existing tables with network_id
        conn.execute(text("""
            UPDATE persons SET network_id = :network_id
        """), {"network_id": network_uuid})

        conn.execute(text("""
            UPDATE interactions SET network_id = :network_id
        """), {"network_id": network_uuid})

        conn.execute(text("""
            UPDATE conversations SET network_id = :network_id
        """), {"network_id": network_uuid})

        conn.execute(text("""
            UPDATE actions SET network_id = :network_id
        """), {"network_id": network_uuid})

        conn.commit()

if __name__ == "__main__":
    migrate_legacy_data()
```

### Step 1.3: Add Constraints and Indexes
```sql
-- Make network_id columns NOT NULL
ALTER TABLE persons ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE interactions ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE conversations ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE actions ALTER COLUMN network_id SET NOT NULL;

-- Add foreign key constraints
ALTER TABLE persons ADD CONSTRAINT fk_persons_network
FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE;

-- Add indexes for performance
CREATE INDEX idx_persons_network_id ON persons(network_id);
CREATE INDEX idx_interactions_network_id ON interactions(network_id);
CREATE INDEX idx_conversations_network_id ON conversations(network_id);
CREATE INDEX idx_actions_network_id ON actions(network_id);
```

## Phase 2: API Layer Migration

### Step 2.1: Create New API Structure
```python
# app/api/v1/networks.py
from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_network_auth
from app.services.network_service import NetworkService

router = APIRouter(prefix="/networks", tags=["networks"])

@router.post("/")
async def create_network(network_data: NetworkCreateRequest):
    """Create a new network"""
    return await NetworkService.create_network(network_data)

@router.get("/{network_id}")
async def get_network(
    network_id: str,
    auth: dict = Depends(get_network_auth)
):
    """Get network details"""
    return await NetworkService.get_network(network_id, auth["network_id"])
```

### Step 2.2: Implement Network Service
```python
# app/services/network_service.py
from app.models.network import Network
from app.core.database import get_db
from app.core.auth import generate_network_token

class NetworkService:
    @staticmethod
    async def create_network(network_data: NetworkCreateRequest) -> NetworkResponse:
        """Create a new network"""
        db = next(get_db())

        # Generate unique network ID
        network_id = generate_network_id()

        # Create network record
        network = Network(
            network_id=network_id,
            name=network_data.name,
            description=network_data.description,
            settings=network_data.settings
        )

        db.add(network)
        db.commit()
        db.refresh(network)

        # Generate authentication token
        token = generate_network_token(network_id)

        return NetworkResponse(
            network_id=network_id,
            name=network.name,
            status="active",
            created_at=network.created_at,
            websocket_url=f"ws://api.mira.com/ws/{network_id}",
            client_registration_token=token
        )
```

### Step 2.3: Add Legacy API Compatibility
```python
# app/api/v1/legacy.py
from fastapi import APIRouter, Depends
from app.core.legacy_compat import LegacyCompatService

router = APIRouter(prefix="/legacy", tags=["legacy"])

@router.get("/interactions")
async def get_legacy_interactions(
    limit: int = 50,
    auth: dict = Depends(get_legacy_auth)
):
    """Legacy endpoint for backward compatibility"""
    # Map to new network-based endpoint
    return await LegacyCompatService.get_interactions(
        network_id=auth["legacy_network_id"],
        limit=limit
    )
```

## Phase 3: Audio Processing Migration

### Step 3.1: Create Multi-Stream Audio Processor
```python
# app/services/audio/multi_stream_processor.py
from app.services.audio.quality_scorer import AudioQualityScorer
from app.models.audio import AudioStream, QualityMetrics

class MultiStreamProcessor:
    def __init__(self):
        self.quality_scorer = AudioQualityScorer()
        self.active_streams = {}

    async def process_audio_stream(
        self,
        network_id: str,
        client_id: str,
        audio_data: bytes
    ) -> AudioProcessingResult:
        """Process audio stream and update quality metrics"""

        # Analyze audio quality
        quality_metrics = await self.quality_scorer.analyze(audio_data)

        # Update stream quality
        self.active_streams[f"{network_id}:{client_id}"] = {
            "client_id": client_id,
            "quality_metrics": quality_metrics,
            "last_update": datetime.now()
        }

        # Select best stream for processing
        best_stream = self.get_best_stream(network_id)

        return AudioProcessingResult(
            client_id=client_id,
            quality_score=quality_metrics.overall_score,
            selected_for_processing=(best_stream.client_id == client_id),
            processing_id=generate_processing_id()
        )
```

### Step 3.2: Migrate Audio Processing Pipeline
```python
# app/services/audio/processing_pipeline.py
from app.services.audio.speech_to_text import SpeechToTextProcessor
from app.services.audio.speaker_recognition import SpeakerRecognitionProcessor

class AudioProcessingPipeline:
    def __init__(self):
        self.speech_to_text = SpeechToTextProcessor()
        self.speaker_recognition = SpeakerRecognitionProcessor()
        self.multi_stream_processor = MultiStreamProcessor()

    async def process_audio(
        self,
        network_id: str,
        audio_data: bytes,
        client_id: str
    ) -> Interaction:
        """Process audio through the complete pipeline"""

        # Process through multi-stream processor
        stream_result = await self.multi_stream_processor.process_audio_stream(
            network_id, client_id, audio_data
        )

        # Only process if this is the best stream
        if not stream_result.selected_for_processing:
            return None

        # Speech-to-text processing
        transcription = await self.speech_to_text.transcribe(audio_data)

        # Speaker recognition
        speaker_result = await self.speaker_recognition.process_speaker(
            audio_data, network_id
        )

        # Create interaction
        interaction = Interaction(
            network_id=network_id,
            client_id=client_id,
            speaker_id=speaker_result.speaker_id,
            text=transcription.text,
            voice_embedding=speaker_result.embedding,
            audio_quality_score=stream_result.quality_score
        )

        return interaction
```

## Phase 4: WebSocket Implementation

### Step 4.1: Create WebSocket Manager
```python
# app/services/websocket/manager.py
from fastapi import WebSocket
from typing import Dict, List
import json

class WebSocketManager:
    def __init__(self):
        self.network_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, network_id: str, client_id: str):
        """Handle new WebSocket connection"""
        await websocket.accept()

        if network_id not in self.network_connections:
            self.network_connections[network_id] = []

        self.network_connections[network_id].append(websocket)

        # Send connection confirmation
        await self.send_to_client(websocket, {
            "type": "connection_established",
            "client_id": client_id,
            "network_id": network_id
        })

    async def broadcast_to_network(
        self,
        network_id: str,
        message: dict,
        exclude_client: str = None
    ):
        """Broadcast message to all clients in network"""
        if network_id not in self.network_connections:
            return

        for websocket in self.network_connections[network_id]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                # Remove disconnected clients
                self.network_connections[network_id].remove(websocket)

    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client"""
        await websocket.send_text(json.dumps(message))
```

### Step 4.2: Implement Real-time Notifications
```python
# app/services/notifications/real_time.py
from app.services.websocket.manager import WebSocketManager
from app.models.interaction import Interaction

class RealTimeNotificationService:
    def __init__(self):
        self.ws_manager = WebSocketManager()

    async def notify_interaction_created(
        self,
        network_id: str,
        interaction: Interaction
    ):
        """Notify all clients about new interaction"""
        message = {
            "type": "interaction_created",
            "interaction_id": str(interaction.id),
            "text": interaction.text,
            "speaker_id": str(interaction.speaker_id),
            "client_id": str(interaction.client_id),
            "timestamp": interaction.timestamp.isoformat()
        }

        await self.ws_manager.broadcast_to_network(network_id, message)

    async def notify_audio_stream_selected(
        self,
        network_id: str,
        client_id: str,
        quality_score: float
    ):
        """Notify about audio stream selection"""
        message = {
            "type": "audio_stream_selected",
            "selected_client_id": client_id,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }

        await self.ws_manager.broadcast_to_network(network_id, message)
```

## Phase 5: Legacy System Deprecation

### Step 5.1: Create Migration Tool
```python
# tools/migrate_legacy_networks.py
import asyncio
from app.services.migration.legacy_migrator import LegacyMigrator

async def migrate_legacy_networks():
    """Migrate legacy data to new network structure"""
    migrator = LegacyMigrator()

    # Get all legacy networks
    legacy_networks = await migrator.get_legacy_networks()

    for legacy_network in legacy_networks:
        try:
            # Create new network
            new_network = await migrator.create_network_from_legacy(legacy_network)

            # Migrate data
            await migrator.migrate_network_data(legacy_network.id, new_network.id)

            # Update client configurations
            await migrator.update_client_configurations(new_network.id)

            print(f"Migrated network: {new_network.network_id}")

        except Exception as e:
            print(f"Failed to migrate network {legacy_network.id}: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(migrate_legacy_networks())
```

### Step 5.2: Gradual Legacy Deprecation
```python
# app/core/legacy_deprecation.py
from datetime import datetime, timedelta
from app.core.config import settings

class LegacyDeprecationManager:
    def __init__(self):
        self.deprecation_date = datetime.now() + timedelta(days=90)
        self.warning_period = datetime.now() + timedelta(days=30)

    def is_legacy_deprecated(self) -> bool:
        """Check if legacy system is deprecated"""
        return datetime.now() > self.deprecation_date

    def should_show_deprecation_warning(self) -> bool:
        """Check if deprecation warning should be shown"""
        return datetime.now() > self.warning_period

    def get_deprecation_message(self) -> dict:
        """Get deprecation warning message"""
        return {
            "warning": "Legacy API endpoints are deprecated",
            "deprecation_date": self.deprecation_date.isoformat(),
            "migration_guide": "https://docs.mira.com/migration",
            "new_endpoints": {
                "interactions": "/api/v1/networks/{network_id}/interactions",
                "audio": "/api/v1/networks/{network_id}/audio"
            }
        }
```

## Testing Strategy

### Step 6.1: Create Migration Tests
```python
# tests/test_migration.py
import pytest
from app.services.migration.legacy_migrator import LegacyMigrator

class TestMigration:
    @pytest.fixture
    async def migrator(self):
        return LegacyMigrator()

    async def test_network_creation(self, migrator):
        """Test network creation from legacy data"""
        legacy_network = await migrator.get_legacy_network("legacy_123")
        new_network = await migrator.create_network_from_legacy(legacy_network)

        assert new_network.network_id is not None
        assert new_network.name == legacy_network.name
        assert new_network.status == "active"

    async def test_data_migration(self, migrator):
        """Test data migration between systems"""
        legacy_network_id = "legacy_123"
        new_network_id = "net_abc123def456"

        await migrator.migrate_network_data(legacy_network_id, new_network_id)

        # Verify data was migrated correctly
        interactions = await migrator.get_network_interactions(new_network_id)
        assert len(interactions) > 0

        for interaction in interactions:
            assert interaction.network_id == new_network_id
```

### Step 6.2: Performance Testing
```python
# tests/test_performance.py
import asyncio
import time
from app.services.audio.multi_stream_processor import MultiStreamProcessor

class TestPerformance:
    async def test_audio_processing_performance(self):
        """Test audio processing performance with multiple streams"""
        processor = MultiStreamProcessor()

        # Simulate multiple audio streams
        start_time = time.time()

        tasks = []
        for i in range(10):  # 10 concurrent streams
            task = processor.process_audio_stream(
                f"network_{i}",
                f"client_{i}",
                generate_test_audio()
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Should process 10 streams in under 1 second
        assert (end_time - start_time) < 1.0
        assert len(results) == 10
```

## Rollback Plan

### Step 7.1: Database Rollback
```sql
-- Rollback script: rollback_migration.sql
-- Remove network_id columns (if needed)
ALTER TABLE persons DROP COLUMN network_id;
ALTER TABLE interactions DROP COLUMN network_id;
ALTER TABLE conversations DROP COLUMN network_id;
ALTER TABLE actions DROP COLUMN network_id;

-- Drop new tables
DROP TABLE IF EXISTS clients;
DROP TABLE IF EXISTS networks;
DROP TABLE IF EXISTS calendar_events;
DROP TABLE IF EXISTS reminders;
```

### Step 7.2: Application Rollback
```python
# rollback_application.py
import subprocess
import shutil
from pathlib import Path

def rollback_application():
    """Rollback application to legacy version"""

    # Stop current application
    subprocess.run(["pkill", "-f", "mira"])

    # Restore legacy code
    shutil.copytree("legacy_code", "app", dirs_exist_ok=True)

    # Restore legacy database
    subprocess.run(["cp", "backup/mira.db", "mira.db"])

    # Start legacy application
    subprocess.run(["python", "mira.py"])
```

## Monitoring and Validation

### Step 8.1: Migration Monitoring
```python
# app/services/migration/monitor.py
class MigrationMonitor:
    def __init__(self):
        self.migration_stats = {
            "networks_migrated": 0,
            "interactions_migrated": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

    def record_migration_success(self, network_id: str, interactions_count: int):
        """Record successful migration"""
        self.migration_stats["networks_migrated"] += 1
        self.migration_stats["interactions_migrated"] += interactions_count

    def record_migration_error(self, error: Exception):
        """Record migration error"""
        self.migration_stats["errors"] += 1
        logger.error(f"Migration error: {error}")

    def get_migration_status(self) -> dict:
        """Get current migration status"""
        return {
            **self.migration_stats,
            "duration": (datetime.now() - self.migration_stats["start_time"]).total_seconds(),
            "success_rate": self.migration_stats["networks_migrated"] /
                          (self.migration_stats["networks_migrated"] + self.migration_stats["errors"])
        }
```

### Step 8.2: Data Validation
```python
# tools/validate_migration.py
async def validate_migration():
    """Validate that migration was successful"""

    # Check data integrity
    legacy_count = await count_legacy_interactions()
    new_count = await count_new_interactions()

    assert legacy_count == new_count, f"Count mismatch: {legacy_count} vs {new_count}"

    # Check network isolation
    networks = await get_all_networks()
    for network in networks:
        interactions = await get_network_interactions(network.id)
        for interaction in interactions:
            assert interaction.network_id == network.id

    print("Migration validation successful!")
```

## Timeline

### Week 1-2: Database Migration
- Create new schema
- Migrate existing data
- Add constraints and indexes
- Validate data integrity

### Week 3-4: API Layer Migration
- Implement new API endpoints
- Add network authentication
- Create legacy compatibility layer
- Test API functionality

### Week 5-6: Audio Processing Migration
- Implement multi-stream processor
- Migrate audio processing pipeline
- Add quality scoring
- Test audio processing

### Week 7-8: WebSocket Implementation
- Create WebSocket manager
- Implement real-time notifications
- Add client connection management
- Test real-time features

### Week 9-10: Testing and Validation
- Comprehensive testing
- Performance testing
- Data validation
- Bug fixes and optimization

### Week 11-12: Legacy Deprecation
- Gradual deprecation
- Client migration
- Legacy system shutdown
- Final cleanup

This migration plan ensures a smooth transition from the legacy system to the new network-based architecture while maintaining data integrity and minimizing downtime.



