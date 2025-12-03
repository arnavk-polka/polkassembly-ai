# Polkadot AI Chatbot

Live bot: https://klara.polkassembly.io

An AI-powered chatbot system that provides intelligent answers about the Polkadot ecosystem using RAG (Retrieval-Augmented Generation) with OpenAI embeddings, ChromaDB, and dynamic SQL generation for on-chain data queries.

## Features

- 📚 **Knowledge Base**: Built from Polkadot Wiki and Forum data
- 🔍 **Semantic Search**: Uses OpenAI embeddings for accurate document retrieval
- 🤖 **AI-Powered Answers**: Generates contextual responses using GPT and Gemini models
- 🗄️ **Dynamic SQL Generation**: Uses Gemini 2.5 Pro for intelligent SQL query generation from natural language
- 📊 **On-Chain Data Access**: Direct querying of governance and voting data from PostgreSQL
- 🚀 **Fast API**: RESTful API with automatic documentation
- 💾 **Persistent Storage**: ChromaDB for efficient vector storage
- 🔧 **Configurable**: Environment-based configuration
- 📊 **Analytics**: Built-in statistics and monitoring
- 🚀 **Production Ready**: Gunicorn with multiple workers for scalability
- 🔄 **Easy Management**: Startup/stop scripts for complete system management
- 📝 **Comprehensive Logging**: Separate log files for each service component


## Quick Start

For detailed installation instructions, see [setup.md](setup.md).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp env.example .env
# Edit .env with your API keys and data paths

# 3. Create embeddings (see setup.md for details)
python src/ingestion/embeddings/create_static_embeddings.py
python src/ingestion/embeddings/index_dynamic_embeddings.py

# 4. Start the API server
python run_server.py
```



## API Endpoints

### 🏥 Health Check
```http
GET /health
```

### 🤖 Ask Questions
```http
POST /query
Content-Type: application/json

{
  "question": "What is Polkadot?",
  "user_id": "krishna",
  "client_ip": "192.168.1.1",
  "max_chunks": 5,
  "include_sources": true,
  "custom_prompt": "optional custom system prompt"
}
```

**Response:**
```json
{
  "answer": "Polkadot is a blockchain protocol...",
  "sources": [...],
  "follow_up_questions": ["...", "...", "..."],
  "remaining_requests": 7
}
```

### 🔍 Search Documents
```http
POST /search
Content-Type: application/json

{
  "query": "staking rewards",
  "n_results": 10,
  "source_filter": "polkadot_wiki"
}
```

### 📊 Get Statistics
```http
GET /stats
```

### 🔐 Rate Limiting

The API implements Redis-based rate limiting to prevent abuse:

- **Default Limits**: 20 requests per hour per user_id
- **Rate Limit Response**: HTTP 429 when limit exceeded
- **Per-User Tracking**: Each user_id has separate rate limits
- **Automatic Reset**: Limits reset after time window expires

**Check Rate Limit Status:**
```http
GET /rate-limit/{user_id}
```

**Response:**
```json
{
  "user_id": "krishna",
  "rate_limit_stats": {
    "max_requests": 20,
    "used_requests": 13,
    "remaining_requests": 7,
    "time_window_seconds": 3600
  }
}
```

## Usage Examples

### Python Client Example

```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/query", json={
    "question": "How do I stake DOT tokens?",
    "user_id": "test_user",
    "client_ip": "192.168.1.1",
    "max_chunks": 3,
    "include_sources": True
})

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Remaining requests: {data['remaining_requests']}")
print(f"Sources: {len(data['sources'])}")
print(f"Follow-up questions: {data['follow_up_questions']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are parachains?",
    "user_id": "curl_user",
    "client_ip": "192.168.1.1",
    "max_chunks": 3,
    "include_sources": true
  }'

# Search documents
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "governance voting",
    "n_results": 5
  }'
```

## Key Configuration

Main configuration options (see [setup.md](setup.md) for complete list):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `GEMINI_API_KEY` | Your Google Gemini API key (required for SQL generation) |
| `STATIC_DATA_PATH` | Path to static documentation files |
| `POSTGRES_HOST` | PostgreSQL host for governance data |
| `POSTGRES_DATABASE` | PostgreSQL database name |
| `POSTGRES_USER` | PostgreSQL username |
| `POSTGRES_PASSWORD` | PostgreSQL password |

## API Documentation

Refer to your deployment’s API host for documentation endpoints if enabled.

## Contributing

1. Fork the repository
2. Create a feature branch  
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For detailed setup, troubleshooting, and development information, see [setup.md](setup.md).

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check logs for error details