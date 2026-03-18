# RAG App — Claude Code Context

## What this project is
A Retrieval-Augmented Generation (RAG) REST API built in Java.
Users upload `.txt` documents, ask questions, and get answers grounded in those documents.
If no relevant documents are found, the app falls back to Claude's general knowledge.

## Stack
- **Java 21+** (tested on Java 25)
- **Spring Boot 3.2.3** — REST API, embedded Tomcat
- **Apache Lucene 9.10** — on-disk vector index with HNSW similarity search
- **Voyage AI** (`voyage-3`) — generates 1024-dimensional embeddings
- **Anthropic Claude** (`claude-sonnet-4-20250514`) — answer generation
- **Springdoc OpenAPI 2.3.0** — Swagger UI at `/swagger-ui.html`
- **JUnit 5 + Spring Boot Test** — unit and integration tests

## Project structure
```
src/
  main/java/com/example/rag/
    RagApp.java              ← entire application in one file
  main/resources/
    application.properties
  test/java/com/example/rag/
    RagAppTest.java          ← unit + integration tests
pom.xml
README.md
CLAUDE.md
```

## How to run
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export VOYAGE_API_KEY=pa-...
mvn spring-boot:run
```

## How to test
```bash
mvn clean test
```

## How to re-index documents
```bash
rm -rf lucene-index/
mvn spring-boot:run
```

## API endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET  | `/` | Health check |
| POST | `/documents` | Upload and index a .txt file |
| POST | `/ask` | Ask a question |
| GET  | `/conversations/{id}` | Get a specific conversation |
| GET  | `/conversations?page=0&size=10` | List all conversations, paginated |

## Key design decisions and why

### Single-file application (RagApp.java)
Everything lives in one file intentionally — this is a portfolio/learning project.
Do not split into multiple files unless the project grows significantly.

### HttpSender interface
```java
public interface HttpSender {
    String post(String url, String body, Map<String, String> headers) throws Exception;
}
```
This thin interface wraps ALL outbound HTTP calls (both Voyage AI and Claude API).
It exists solely to make testing possible — `java.net.http.HttpClient` is a sealed
JDK class that cannot be mocked by Mockito or proxied on Java 21+.
In tests, `httpSender` is replaced via reflection with a lambda returning hardcoded responses.
Do NOT remove this interface or inline the HttpClient calls.

### Lucene for vector storage (not a vector DB)
We use Lucene's `KnnFloatVectorField` with `VectorSimilarityFunction.DOT_PRODUCT`.
Voyage AI normalizes vectors, so DOT_PRODUCT = cosine similarity.
The index is stored on disk at `lucene-index/` and persists between restarts.
`IndexWriterConfig.OpenMode.CREATE_OR_APPEND` means new documents are added
without wiping the existing index.

### Embedding dimensions
voyage-3 produces 1024-dimensional vectors. If switching models:
- voyage-3-lite → 512 dimensions
- voyage-3-large → 2048 dimensions
The dimension is baked into `KnnFloatVectorField` at index time — changing it
requires deleting `lucene-index/` and re-indexing.

### Relevance threshold
`RELEVANCE_THRESHOLD = 0.7f` — chunks scoring below this are ignored and
the app falls back to Claude's general knowledge.
Tune this if retrieval is too strict (lower) or too permissive (higher).

### Conversation persistence
Each conversation is a JSON file: `conversations/{uuid}.json`
Format:
```json
{
  "conversationId": "abc-123",
  "updatedAt": "2026-03-18T22:00:00Z",
  "messages": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```
The full message history is sent to Claude on every follow-up — this is what
enables multi-turn conversations. Conversations are sorted by file modification
time (most recent first) in the paginated list endpoint.

### System prompt override
`POST /ask` accepts an optional `systemPrompt` field that overrides the default
Claude instructions for that request. The retrieved context is always appended
automatically. This lets callers change how Claude reasons without modifying code.

### Chunking strategy
Documents are split into chunks of 400 characters with 80-character overlap.
Overlap prevents sentences from being cut off at chunk boundaries.
Each chunk is embedded separately via Voyage AI and stored as a Lucene document.

## Auto-created folders (gitignored)
- `data/` — where uploaded .txt files are stored
- `lucene-index/` — Lucene on-disk vector index
- `conversations/` — conversation history as JSON files

All three are auto-created on startup if missing. Safe to delete and recreate.

## Environment variables
| Variable | Used for |
|----------|----------|
| `ANTHROPIC_API_KEY` | Claude API (generation) |
| `VOYAGE_API_KEY` | Voyage AI API (embeddings) |

Both are read via `System.getenv()`. The app throws `IllegalStateException` at
startup if either is missing.

## Known issues / gotchas
- **Java compiler version**: `pom.xml` sets `maven.compiler.release=21`. If Maven
  compiles at a lower version, text blocks and records will fail. Always use
  `mvn clean` before `mvn test` after changing the pom.
- **Re-indexing**: Adding files to `data/` manually after the index is built will
  NOT automatically index them. Use `POST /documents` or delete `lucene-index/`
  and restart.
- **Duplicate chunks**: Uploading the same file twice via `POST /documents` will
  create duplicate chunks in the index. Delete `lucene-index/` and restart to fix.
- **Spring Boot package scanning**: `RagApp.java` must be in package `com.example.rag`.
  Running from the default package causes Spring to scan all of Spring's internal
  classes, triggering R2DBC errors.

## Testing approach
- Unit tests call private methods directly via reflection (`getDeclaredMethod` + `setAccessible`)
- Integration tests use `@SpringBootTest` + `MockMvc`
- `@BeforeEach` replaces `httpSender` via reflection with a fake lambda — no real API calls
- Tests are ordered (`@TestMethodOrder`) because some integration tests share `conversationId`
- Test properties set dummy API keys so `@PostConstruct` validation passes

## What NOT to do
- Do not mock `java.net.http.HttpClient` — it is sealed and unmockable on Java 21+
- Do not use `@MockBean HttpClient` — same reason
- Do not put `RagApp.java` in the default package
- Do not hardcode API keys — always use environment variables
- Do not commit `data/`, `lucene-index/`, `conversations/`, or `target/`

## Development workflow
- **Always update `README.md`** when making significant changes — new endpoints,
  changed run commands, new dependencies, new environment variables, or anything
  a developer cloning the repo would need to know.
- **Always update `CLAUDE.md`** when adding new design decisions, gotchas, or
  architectural changes that future Claude sessions should know about.