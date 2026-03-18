# RAG App

A simple Retrieval-Augmented Generation (RAG) application built in Java using:
- **Apache Lucene** — on-disk vector index with HNSW semantic search
- **Voyage AI** (`voyage-3`) — generates embeddings for documents and questions
- **Anthropic Claude** (`claude-sonnet-4`) — generates answers from retrieved context
- **Spring Boot 3** — REST API layer
- **Springdoc OpenAPI** — Swagger UI for interactive API documentation

When you ask a question, the app searches your documents semantically and passes the most relevant chunks to Claude as context. If no relevant content is found, it falls back to Claude's general knowledge and tells you so.

---

## Project Structure

```
rag-app/
├── src/main/java/com/example/rag/
│   └── RagApp.java          ← entire application, one file
├── src/main/resources/
│   └── application.properties
├── data/                    ← your .txt documents (auto-created, gitignored)
├── lucene-index/            ← on-disk vector index (auto-created, gitignored)
├── conversations/           ← conversation history as JSON (auto-created, gitignored)
└── pom.xml
```

---

## Setup

### 1. Prerequisites
- Java 21+
- Maven 3.8+
- An [Anthropic API key](https://console.anthropic.com)
- A [Voyage AI API key](https://dash.voyageai.com) (free tier available)

### 2. Export API keys
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export VOYAGE_API_KEY=pa-...
```

### 3. Run
```bash
mvn spring-boot:run
```

The API starts at `http://localhost:8080`.
On first run it will index any `.txt` files already in the `data/` folder.

---

## Swagger UI

The easiest way to explore and test the API is via Swagger UI:

```
http://localhost:8080/swagger-ui.html
```

Swagger UI lets you:
- Browse all endpoints with full descriptions
- See request/response schemas and field-level documentation
- Execute real API calls directly from the browser using **Try it out**
- Use pre-built example payloads for common scenarios (simple question, follow-up, custom prompt)

The raw OpenAPI spec (JSON) is also available at:
```
http://localhost:8080/v3/api-docs
```

---

## API Reference

### Health check
```bash
curl http://localhost:8080/
```
```json
{ "status": "ok", "message": "RAG API is running", "version": "1.0" }
```

---

### Upload a document
Saves the file to `data/` and indexes it immediately — no restart needed.
```bash
curl -X POST http://localhost:8080/documents \
  -F "file=@/path/to/your/file.txt"
```
```json
{ "filename": "facts.txt", "chunks": 3 }
```

---

### Ask a question
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of Golustaan?"}'
```
```json
{
  "conversationId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "answer": "Based on the context, PogaNagar is likely the capital...",
  "source": "📄 Answered from: facts.txt"
}
```

If no relevant documents are found, Claude answers from general knowledge:
```json
{
  "conversationId": "f47ac10b-...",
  "answer": "The capital of France is Paris...",
  "source": "🌐 Answered from: Claude's general knowledge (no relevant documents found)"
}
```

---

### Continue a conversation
Pass the `conversationId` from a previous response to continue the same session.
Claude will remember the full conversation history.
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who rules it?",
    "conversationId": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
  }'
```

---

### Override the system prompt
By default Claude is instructed to answer from context and use reasonable inference.
You can override this per-request to change how Claude reasons:
```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of Golustaan?",
    "systemPrompt": "A capital city is where parliament, courts and diplomatic activity are located — not just where a ruler is visited. Use this definition when inferring. Answer using the context below."
  }'
```

---

### Get a specific conversation
```bash
curl http://localhost:8080/conversations/f47ac10b-58cc-4372-a567-0e02b2c3d479
```
```json
{
  "conversationId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "updatedAt": "2026-03-18T22:00:00Z",
  "messages": [
    { "role": "user",      "content": "What is the capital of Golustaan?" },
    { "role": "assistant", "content": "Based on the context, PogaNagar..." },
    { "role": "user",      "content": "Who rules it?" },
    { "role": "assistant", "content": "Golustaan is ruled by a King..." }
  ]
}
```

---

### List all conversations (paginated)
Returns all conversations, most recent first.
```bash
curl "http://localhost:8080/conversations?page=0&size=10"
```
```json
{
  "conversations": [ ... ],
  "page": 0,
  "size": 10,
  "total": 42,
  "totalPages": 5
}
```

---

## How data is stored

### Documents (`data/`)
Plain `.txt` files. Uploaded via `POST /documents` or placed here manually.
To add a new file manually and have it indexed, either:
- Use `POST /documents` — indexes immediately without restart
- Delete `lucene-index/` and restart — rebuilds the full index from all files in `data/`

### Vector index (`lucene-index/`)
Apache Lucene on-disk index. Each document is split into overlapping chunks of
~400 characters, embedded via Voyage AI into 1024-dimensional vectors, and stored
using Lucene's `KnnFloatVectorField`. Similarity search uses the HNSW algorithm.

Delete this folder to force a full re-index on next startup.

### Conversations (`conversations/`)
One JSON file per conversation, named by UUID. Each file contains the full
message history (both questions and answers) and is updated after every turn.
This history is sent to Claude on every follow-up question, enabling
multi-turn conversations.

Example: `conversations/f47ac10b-58cc-4372-a567-0e02b2c3d479.json`

---

## Re-indexing

After adding new files to `data/` manually:
```bash
rm -rf lucene-index/
mvn spring-boot:run
```

---

## License

MIT