# Database Design for Multi-Network Architecture

## Overview

The database design supports the network-based architecture by implementing proper data isolation, multi-client support, and scalable relationships. All data is scoped to Network IDs to ensure complete separation between different user networks.

## Core Principles

### Network Isolation
- **Primary Key**: All tables include `network_id` as a foreign key
- **Data Scoping**: All queries must filter by Network ID
- **No Cross-Network Access**: Impossible to access data from other networks
- **Audit Trail**: Track all network-level operations

### Multi-Client Support
- **Client Tracking**: All interactions linked to specific clients
- **Quality Metrics**: Store audio quality scores per client
- **Connection State**: Track client connection status and capabilities

### Scalability
- **Indexing Strategy**: Optimized indexes for Network ID queries
- **Partitioning**: Consider table partitioning by Network ID for large scale
- **Caching**: Support for application-level caching

## Database Schema

### Networks Table
```sql
CREATE TABLE networks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    settings JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_networks_network_id ON networks(network_id);
CREATE INDEX idx_networks_status ON networks(status);
CREATE INDEX idx_networks_last_activity ON networks(last_activity);
```

### Clients Table
```sql
CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    client_id VARCHAR(50) NOT NULL,
    device_type VARCHAR(50) NOT NULL,
    device_info JSONB DEFAULT '{}',
    capabilities JSONB DEFAULT '[]',
    connection_status VARCHAR(20) DEFAULT 'disconnected' CHECK (connection_status IN ('connected', 'disconnected', 'error')),
    last_seen TIMESTAMP WITH TIME ZONE,
    audio_quality_score FLOAT DEFAULT 0.0,
    location JSONB,
    rssi FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(network_id, client_id)
);

CREATE INDEX idx_clients_network_id ON clients(network_id);
CREATE INDEX idx_clients_client_id ON clients(client_id);
CREATE INDEX idx_clients_connection_status ON clients(connection_status);
CREATE INDEX idx_clients_last_seen ON clients(last_seen);
CREATE INDEX idx_clients_audio_quality ON clients(audio_quality_score);
```

### Persons Table (Updated for Multi-Network)
```sql
CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    name VARCHAR(255),
    index INTEGER NOT NULL,
    voice_embedding JSONB,
    cluster_id INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(network_id, index)
);

CREATE INDEX idx_persons_network_id ON persons(network_id);
CREATE INDEX idx_persons_index ON persons(network_id, index);
CREATE INDEX idx_persons_cluster_id ON persons(network_id, cluster_id);
```

### Interactions Table (Updated for Multi-Network)
```sql
CREATE TABLE interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    client_id UUID REFERENCES clients(id) ON DELETE SET NULL,
    speaker_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,

    text TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Audio processing data
    voice_embedding JSONB,
    audio_quality_score FLOAT,
    audio_duration FLOAT,

    -- NLP processing data
    text_embedding JSONB,
    entities JSONB,
    topics JSONB,
    sentiment FLOAT,

    -- Processing metadata
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    processing_metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_interactions_network_id ON interactions(network_id);
CREATE INDEX idx_interactions_client_id ON interactions(client_id);
CREATE INDEX idx_interactions_speaker_id ON interactions(speaker_id);
CREATE INDEX idx_interactions_conversation_id ON interactions(conversation_id);
CREATE INDEX idx_interactions_timestamp ON interactions(network_id, timestamp);
CREATE INDEX idx_interactions_processing_status ON interactions(processing_status);
```

### Conversations Table (Updated for Multi-Network)
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    title VARCHAR(255),
    topic_summary TEXT,
    context_summary TEXT,
    participant_ids JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_interaction_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_conversations_network_id ON conversations(network_id);
CREATE INDEX idx_conversations_last_interaction ON conversations(network_id, last_interaction_at);
```

### Actions Table (Updated for Multi-Network)
```sql
CREATE TABLE actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,

    action_type VARCHAR(50) NOT NULL,
    action_data JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    result JSONB,
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_actions_network_id ON actions(network_id);
CREATE INDEX idx_actions_interaction_id ON actions(interaction_id);
CREATE INDEX idx_actions_status ON actions(status);
CREATE INDEX idx_actions_type ON actions(action_type);
```

### Calendar Events Table
```sql
CREATE TABLE calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    interaction_id UUID REFERENCES interactions(id) ON DELETE SET NULL,

    title VARCHAR(255) NOT NULL,
    description TEXT,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    location VARCHAR(255),
    attendees JSONB DEFAULT '[]',
    reminder_settings JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calendar_events_network_id ON calendar_events(network_id);
CREATE INDEX idx_calendar_events_start_time ON calendar_events(network_id, start_time);
CREATE INDEX idx_calendar_events_interaction_id ON calendar_events(interaction_id);
```

### Reminders Table
```sql
CREATE TABLE reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    interaction_id UUID REFERENCES interactions(id) ON DELETE SET NULL,

    title VARCHAR(255) NOT NULL,
    description TEXT,
    due_time TIMESTAMP WITH TIME ZONE NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'cancelled', 'overdue')),
    reminder_settings JSONB DEFAULT '{}',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_reminders_network_id ON reminders(network_id);
CREATE INDEX idx_reminders_due_time ON reminders(network_id, due_time);
CREATE INDEX idx_reminders_status ON reminders(status);
CREATE INDEX idx_reminders_interaction_id ON reminders(interaction_id);
```

### Network Settings Table
```sql
CREATE TABLE network_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,

    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB NOT NULL,
    setting_type VARCHAR(50) DEFAULT 'string' CHECK (setting_type IN ('string', 'number', 'boolean', 'json', 'array')),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(network_id, setting_key)
);

CREATE INDEX idx_network_settings_network_id ON network_settings(network_id);
CREATE INDEX idx_network_settings_key ON network_settings(setting_key);
```

### Client Sessions Table
```sql
CREATE TABLE client_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    network_id UUID NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,

    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    capabilities JSONB DEFAULT '[]',

    connected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    disconnected_at TIMESTAMP WITH TIME ZONE,

    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_client_sessions_network_id ON client_sessions(network_id);
CREATE INDEX idx_client_sessions_client_id ON client_sessions(client_id);
CREATE INDEX idx_client_sessions_token ON client_sessions(session_token);
CREATE INDEX idx_client_sessions_active ON client_sessions(is_active);
```

## Data Access Patterns

### Network-Scoped Queries
All queries must include Network ID filtering:

```sql
-- Get all interactions for a network
SELECT * FROM interactions
WHERE network_id = $1
ORDER BY timestamp DESC;

-- Get active clients for a network
SELECT * FROM clients
WHERE network_id = $1
AND connection_status = 'connected';

-- Get conversations with recent activity
SELECT * FROM conversations
WHERE network_id = $1
AND last_interaction_at > NOW() - INTERVAL '7 days'
ORDER BY last_interaction_at DESC;
```

### Client-Scoped Queries
Queries can be further scoped to specific clients:

```sql
-- Get interactions from a specific client
SELECT * FROM interactions
WHERE network_id = $1
AND client_id = $2
ORDER BY timestamp DESC;

-- Get client's audio quality history
SELECT timestamp, audio_quality_score
FROM interactions
WHERE network_id = $1
AND client_id = $2
AND audio_quality_score IS NOT NULL
ORDER BY timestamp DESC;
```

## Migration from Legacy Schema

### Step 1: Add Network Support
```sql
-- Add network_id columns to existing tables
ALTER TABLE persons ADD COLUMN network_id UUID;
ALTER TABLE interactions ADD COLUMN network_id UUID;
ALTER TABLE conversations ADD COLUMN network_id UUID;
ALTER TABLE actions ADD COLUMN network_id UUID;

-- Create default network for existing data
INSERT INTO networks (network_id, name, status)
VALUES ('legacy_migration', 'Legacy Data Migration', 'active');

-- Update existing records with default network
UPDATE persons SET network_id = (SELECT id FROM networks WHERE network_id = 'legacy_migration');
UPDATE interactions SET network_id = (SELECT id FROM networks WHERE network_id = 'legacy_migration');
UPDATE conversations SET network_id = (SELECT id FROM networks WHERE network_id = 'legacy_migration');
UPDATE actions SET network_id = (SELECT id FROM networks WHERE network_id = 'legacy_migration');

-- Make network_id NOT NULL
ALTER TABLE persons ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE interactions ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE conversations ALTER COLUMN network_id SET NOT NULL;
ALTER TABLE actions ALTER COLUMN network_id SET NOT NULL;
```

### Step 2: Add Foreign Key Constraints
```sql
-- Add foreign key constraints
ALTER TABLE persons ADD CONSTRAINT fk_persons_network
FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE;

ALTER TABLE interactions ADD CONSTRAINT fk_interactions_network
FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE;

ALTER TABLE conversations ADD CONSTRAINT fk_conversations_network
FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE;

ALTER TABLE actions ADD CONSTRAINT fk_actions_network
FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE;
```

### Step 3: Add New Tables
```sql
-- Create new tables for multi-network support
-- (Use the CREATE TABLE statements from above)
```

## Performance Optimization

### Indexing Strategy
- **Primary Indexes**: Network ID on all tables
- **Composite Indexes**: Network ID + timestamp for time-based queries
- **Partial Indexes**: Active clients, recent interactions
- **Covering Indexes**: Include frequently accessed columns

### Query Optimization
- **Network Filtering**: Always filter by Network ID first
- **Limit Clauses**: Use appropriate LIMIT clauses for pagination
- **Join Optimization**: Use appropriate JOIN strategies
- **Query Caching**: Cache frequently accessed data

### Partitioning Strategy
For large-scale deployments, consider partitioning by Network ID:

```sql
-- Example partitioning for interactions table
CREATE TABLE interactions_2024_01 PARTITION OF interactions
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Partition by network_id for very large datasets
CREATE TABLE interactions_net_1 PARTITION OF interactions
FOR VALUES IN ('net_abc123def456');
```

## Data Retention and Cleanup

### Retention Policies
- **Interactions**: Keep for 2 years, archive older data
- **Client Sessions**: Keep for 30 days after disconnection
- **Audio Data**: Keep for 90 days, then compress/archive
- **Logs**: Keep for 1 year, then archive

### Cleanup Procedures
```sql
-- Archive old interactions
INSERT INTO interactions_archive
SELECT * FROM interactions
WHERE created_at < NOW() - INTERVAL '2 years';

-- Clean up old client sessions
DELETE FROM client_sessions
WHERE disconnected_at < NOW() - INTERVAL '30 days'
AND is_active = FALSE;

-- Update network activity timestamps
UPDATE networks
SET last_activity = (
    SELECT MAX(timestamp)
    FROM interactions
    WHERE network_id = networks.id
);
```

This database design provides a robust foundation for the multi-network Mira system while maintaining data integrity, performance, and scalability.



