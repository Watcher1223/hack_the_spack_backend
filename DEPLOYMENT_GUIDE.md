# Deployment Guide - Universal Adapter API

**Version:** 2.0.0
**Date:** 2026-01-31
**Status:** Production Ready

---

## Quick Start with Docker Compose

### 1. Environment Setup

Create `dev.env` file:

```bash
# Required
VOYAGE_API_KEY=your_voyage_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional (docker-compose sets this automatically)
# MONGODB_URI=mongodb://admin:admin123@mongodb:27017/agent_db?authSource=admin
```

### 2. Start Services

```bash
# Start MongoDB + API
docker-compose up -d

# View logs
docker-compose logs -f agent

# Check status
docker-compose ps
```

### 3. Verify Deployment

```bash
# Health check
curl http://localhost:8001/health

# Should return:
# {"status":"healthy","service":"universal-adapter-api","version":"2.0.0"}

# Test chat endpoint
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello!"}'

# Test tools endpoint
curl http://localhost:8001/tools

# Test discovery stream
curl -N http://localhost:8001/api/discovery/stream
```

---

## Architecture

```
┌─────────────────────────────────────┐
│     Docker Compose Services         │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │   MongoDB    │  │  API Server │ │
│  │   (Port      │  │  (Port      │ │
│  │   27017)     │◄─┤  8001)      │ │
│  └──────────────┘  └─────────────┘ │
│                                     │
└─────────────────────────────────────┘
         ▲
         │ HTTP
         │
    ┌────┴────┐
    │   UI    │
    │ (3000)  │
    └─────────┘
```

---

## Services

### MongoDB
- **Image:** mongo:8.0
- **Port:** 27017
- **Credentials:** admin/admin123
- **Database:** agent_db
- **Health Check:** mongosh ping
- **Data Volume:** Persisted in `mongodb_data`

### API Server
- **Build:** From Dockerfile (Python 3.13)
- **Port:** 8001
- **Environment:** dev.env
- **Volumes:**
  - `./artifacts` → `/app/artifacts` (file operations)
  - `./logs` → `/app/logs` (logging)
- **Health Check:** curl to `/health` endpoint
- **Depends On:** MongoDB (waits for healthy status)

---

## API Endpoints

All endpoints available at `http://localhost:8001`

### P0 Critical
- `POST /chat` - Chat with workflow tracking
- `GET /api/discovery/stream` - SSE discovery logs
- `GET /tools` - List tools with enhanced metadata
- `POST /api/forge/generate` - Generate tools from API docs
- `POST /tools/{name}/execute` - Execute tool with metadata

### P1 Important
- `GET /api/actions` - Action feed
- `GET /api/governance/verified-tools` - Verified tools

### Legacy
- `GET /` - API info
- `GET /health` - Health check
- `GET /tools/search` - Semantic search
- `GET /tools/{name}` - Get tool
- `DELETE /tools/{name}` - Delete tool
- `GET /conversations` - List conversations
- `GET /conversations/{id}` - Get conversation

---

## Configuration

### Environment Variables

**Required:**
- `VOYAGE_API_KEY` - Voyage AI embeddings
- `FIRECRAWL_API_KEY` - Web scraping
- `OPENROUTER_API_KEY` - LLM access

**Auto-configured by docker-compose:**
- `PORT=8001`
- `MONGODB_URI=mongodb://admin:admin123@mongodb:27017/agent_db?authSource=admin`

### CORS

Currently allows all origins (`*`). For production, update `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Development Workflow

### Local Development (Without Docker)

```bash
# 1. Start MongoDB
docker-compose up -d mongodb

# 2. Install dependencies
uv sync

# 3. Set environment
cp .env.example dev.env
# Edit dev.env with your keys

# 4. Run server
uv run server.py

# Server runs on http://localhost:8001
```

### With Docker (Production-like)

```bash
# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f agent

# Restart service
docker-compose restart agent

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Database

### MongoDB Collections

**tools** - Tool marketplace
```javascript
{
  name: "get_crypto_price",
  description: "Get cryptocurrency price",
  status: "PROD-READY",
  category: "crypto",
  tags: ["crypto", "price"],
  verified: true,
  usage_count: 0,
  parameters: {...},
  code: "async def get_crypto_price...",
  embedding: [1024 dimensions],
  created_at: ISODate(...)
}
```

**conversations** - Chat history
```javascript
{
  conversation_id: "uuid",
  messages: [...],
  tool_calls: [...],
  created_at: ISODate(...)
}
```

### Migrations

Run migration to update existing tools with new fields:

```bash
# Local
uv run migrate_tools_schema.py

# Docker
docker-compose exec agent uv run migrate_tools_schema.py
```

---

## Monitoring

### Logs

```bash
# All logs
docker-compose logs -f

# API only
docker-compose logs -f agent

# MongoDB only
docker-compose logs -f mongodb

# Tail last 100 lines
docker-compose logs --tail=100 agent
```

### Health Checks

```bash
# API health
curl http://localhost:8001/health

# MongoDB health
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
```

### Metrics

Check service stats:
```bash
docker stats agent-mongodb agent
```

---

## Troubleshooting

### API won't start

1. **Check MongoDB is healthy:**
   ```bash
   docker-compose ps mongodb
   # Should show "healthy"
   ```

2. **Check environment variables:**
   ```bash
   docker-compose exec agent env | grep API_KEY
   ```

3. **View startup logs:**
   ```bash
   docker-compose logs agent
   ```

### Database connection errors

1. **Check MongoDB is running:**
   ```bash
   docker-compose ps mongodb
   ```

2. **Test connection:**
   ```bash
   docker-compose exec mongodb mongosh \
     -u admin -p admin123 --authenticationDatabase admin
   ```

3. **Restart both services:**
   ```bash
   docker-compose restart mongodb agent
   ```

### Port already in use

```bash
# Check what's using port 8001
lsof -i :8001

# Kill process
kill -9 <PID>

# Or use different port
# Edit docker-compose.yml: ports: ["8002:8001"]
```

### Missing tools

Run migration:
```bash
docker-compose exec agent uv run migrate_tools_schema.py
```

---

## Production Deployment

### Checklist

- [ ] Set specific CORS origins (not `*`)
- [ ] Use production MongoDB (not localhost)
- [ ] Set strong MongoDB credentials
- [ ] Enable HTTPS/TLS
- [ ] Add authentication/API keys
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure logging to external service
- [ ] Set up backup for MongoDB
- [ ] Use production LLM API keys with rate limits
- [ ] Enable rate limiting on endpoints
- [ ] Add request validation
- [ ] Set up health checks in load balancer

### Production docker-compose.yml

```yaml
services:
  mongodb:
    # Use MongoDB Atlas or managed MongoDB
    # Or secure local instance with SSL/TLS

  agent:
    restart: always
    environment:
      - PORT=8001
      - MONGODB_URI=${MONGODB_URI}  # From secrets
      - VOYAGE_API_KEY=${VOYAGE_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    # Add reverse proxy (nginx) for HTTPS
```

### Environment Variables (Production)

```bash
# Use secrets management (AWS Secrets Manager, etc.)
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/agent_db
VOYAGE_API_KEY=<production-key>
FIRECRAWL_API_KEY=<production-key>
OPENROUTER_API_KEY=<production-key>
```

---

## Testing

### Integration Tests

```bash
# Start services
docker-compose up -d

# Run tests
pytest tests/integration/

# Or with docker
docker-compose exec agent pytest tests/
```

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Test chat endpoint
ab -n 100 -c 10 -p chat_request.json -T application/json \
  http://localhost:8001/chat

# Test tools endpoint
ab -n 1000 -c 50 http://localhost:8001/tools
```

---

## Backup & Recovery

### Backup MongoDB

```bash
# Backup
docker-compose exec mongodb mongodump \
  --uri="mongodb://admin:admin123@localhost:27017/agent_db?authSource=admin" \
  --out=/backup

# Copy backup out of container
docker cp agent-mongodb:/backup ./mongodb-backup-$(date +%Y%m%d)
```

### Restore MongoDB

```bash
# Copy backup to container
docker cp ./mongodb-backup agent-mongodb:/restore

# Restore
docker-compose exec mongodb mongorestore \
  --uri="mongodb://admin:admin123@localhost:27017/agent_db?authSource=admin" \
  /restore/agent_db
```

---

## Scaling

### Horizontal Scaling

1. Use load balancer (nginx, HAProxy)
2. Run multiple API containers
3. Use shared MongoDB (Atlas or cluster)
4. Session affinity for SSE endpoints

```yaml
# docker-compose-scale.yml
services:
  agent:
    deploy:
      replicas: 3
```

### Vertical Scaling

```yaml
# Increase resources
services:
  agent:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## Support

- **Documentation:** See `UI_INTEGRATION_GUIDE.md`, `API_DOCUMENTATION.md`
- **Issues:** GitHub Issues
- **Logs:** `docker-compose logs -f agent`
- **Health:** `http://localhost:8001/health`

---

**Version:** 2.0.0 | **Status:** Production Ready | **Updated:** 2026-01-31
