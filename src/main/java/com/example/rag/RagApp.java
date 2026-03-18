package com.example.rag;

/**
 * RAG Application with REST APIs
 * Uses Spring Boot 3 for REST + Apache Lucene (on-disk) + Voyage AI embeddings + Claude API.
 *
 * APIs:
 *   GET    /                                 — health check
 *   POST   /documents                        — upload a file, store in data/, index it
 *   POST   /ask                              — ask a question (with optional conversationId and systemPrompt)
 *   GET    /conversations/{conversationId}   — get a specific conversation
 *   GET    /conversations?page=0&size=10     — get all conversations, paginated, most recent first
 *
 * Swagger UI: http://localhost:8080/swagger-ui.html
 *
 * Run:
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   export VOYAGE_API_KEY=pa-...
 *   mvn spring-boot:run
 */

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.FSDirectory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import jakarta.annotation.PostConstruct;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.stream.Stream;

@SpringBootApplication
@RestController
@Tag(name = "RAG API", description = "Retrieval-Augmented Generation using Lucene + Voyage AI + Claude")
public class RagApp {

    // ── 1. Configuration ──────────────────────────────────────────────────────

    private static final String API_KEY             = System.getenv("ANTHROPIC_API_KEY");
    private static final String VOYAGE_KEY          = System.getenv("VOYAGE_API_KEY");
    private static final String CLAUDE_MODEL        = "claude-sonnet-4-20250514";
    private static final String VOYAGE_MODEL        = "voyage-3";
    private static final float  RELEVANCE_THRESHOLD = 0.7f;
    private static final int    TOP_K               = 3;
    private static final int    CHUNK_SIZE          = 400;
    private static final int    OVERLAP             = 80;

    private static final String FIELD_VECTOR  = "vector";
    private static final String FIELD_SOURCE  = "source";
    private static final String FIELD_CONTENT = "content";

    private static final Path DATA_DIR          = Path.of("data");
    private static final Path INDEX_DIR         = Path.of("lucene-index");
    private static final Path CONVERSATIONS_DIR = Path.of("conversations");

    private static final String DEFAULT_RAG_PROMPT = """
        You are a helpful assistant. Answer the user's question using the context below.
        Use reasonable inference when the answer is implied but not explicitly stated.
        If you are inferring rather than quoting directly, clearly indicate this.
        If the context contains no relevant information at all, say so.
        """;

    private static final String DEFAULT_FALLBACK_PROMPT =
        "You are a helpful assistant. Answer questions using your general knowledge.";

    // ── 2. HttpSender interface ───────────────────────────────────────────────

    /**
     * Thin interface wrapping HTTP POST calls.
     * The real implementation uses Java's HttpClient.
     * Tests inject a fake implementation that returns hardcoded responses —
     * this avoids the sealed-class / Mockito / Proxy limitation on Java 21+.
     */
    public interface HttpSender {
        String post(String url, String body, Map<String, String> headers) throws Exception;
    }

    // ── 3. Infrastructure ─────────────────────────────────────────────────────

    private FSDirectory      index;
    private StandardAnalyzer analyzer;
    private ObjectMapper     json;

    // HttpSender is package-private so tests can replace it via reflection
    HttpSender httpSender;

    // ── 4. Request / Response records ─────────────────────────────────────────

    @Schema(description = "Request body for asking a question")
    record AskRequest(
        @Schema(description = "The question to ask", example = "What is the capital of Golustaan?", requiredMode = Schema.RequiredMode.REQUIRED)
        @JsonProperty("question")       String question,

        @Schema(description = "Optional conversation ID to continue an existing session.")
        @JsonProperty("conversationId") String conversationId,

        @Schema(description = "Optional system prompt to override Claude's default instructions.")
        @JsonProperty("systemPrompt")   String systemPrompt
    ) {}

    @Schema(description = "Response from the ask endpoint")
    record AskResponse(
        @Schema(description = "The conversation ID — use this to continue the session")
        @JsonProperty("conversationId") String conversationId,

        @Schema(description = "Claude's answer")
        @JsonProperty("answer")         String answer,

        @Schema(description = "Indicates whether the answer came from your documents or Claude's general knowledge")
        @JsonProperty("source")         String source
    ) {}

    @Schema(description = "Response from the document upload endpoint")
    record IndexResponse(
        @Schema(description = "The filename that was indexed")
        @JsonProperty("filename") String filename,

        @Schema(description = "Number of chunks the document was split into")
        @JsonProperty("chunks")   int chunks
    ) {}

    @Schema(description = "Paginated list of conversations")
    record ConversationsPage(
        @Schema(description = "List of conversations on this page")
        @JsonProperty("conversations") List<Map<String, Object>> conversations,

        @Schema(description = "Current page number (0-based)")
        @JsonProperty("page")          int page,

        @Schema(description = "Number of items per page")
        @JsonProperty("size")          int size,

        @Schema(description = "Total number of conversations")
        @JsonProperty("total")         int total,

        @Schema(description = "Total number of pages")
        @JsonProperty("totalPages")    int totalPages
    ) {}

    // ── 5. Swagger / OpenAPI config ───────────────────────────────────────────

    @Bean
    public OpenAPI openApiConfig() {
        return new OpenAPI()
            .info(new Info()
                .title("RAG API")
                .version("1.0")
                .description("""
                    Retrieval-Augmented Generation API powered by:
                    - **Apache Lucene** — on-disk vector index with HNSW semantic search
                    - **Voyage AI** (voyage-3) — 1024-dimensional text embeddings
                    - **Anthropic Claude** — answer generation with fallback to general knowledge
                    """)
            );
    }

    // ── 6. Startup ────────────────────────────────────────────────────────────

    @PostConstruct
    public void init() throws Exception {
        if (API_KEY == null || API_KEY.isBlank()) {
            throw new IllegalStateException("ANTHROPIC_API_KEY environment variable is not set.");
        }
        if (VOYAGE_KEY == null || VOYAGE_KEY.isBlank()) {
            throw new IllegalStateException("VOYAGE_API_KEY environment variable is not set.");
        }

        json     = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        analyzer = new StandardAnalyzer();

        Files.createDirectories(DATA_DIR);
        Files.createDirectories(INDEX_DIR);
        Files.createDirectories(CONVERSATIONS_DIR);

        this.index = FSDirectory.open(INDEX_DIR);

        // Default HttpSender uses Java's built-in HttpClient
        this.httpSender = (url, body, headers) -> {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .POST(HttpRequest.BodyPublishers.ofString(body));
            headers.forEach(builder::header);
            HttpResponse<String> response = client.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("HTTP error " + response.statusCode() + ": " + response.body());
            }
            return response.body();
        };

        loadDataFolder();

        System.out.println("\n========================================");
        System.out.println("  RAG API  →  http://localhost:8080");
        System.out.println("  Swagger  →  http://localhost:8080/swagger-ui.html");
        System.out.println("========================================\n");
    }

    private void loadDataFolder() throws Exception {
        List<Path> txtFiles;
        try (Stream<Path> stream = Files.list(DATA_DIR)) {
            txtFiles = stream.filter(p -> p.toString().endsWith(".txt")).sorted().toList();
        }

        if (txtFiles.isEmpty()) {
            System.out.println("[i] No .txt files in data/ — upload files via POST /documents.");
            return;
        }

        boolean indexExists = DirectoryReader.indexExists(index);
        if (indexExists) {
            try (DirectoryReader reader = DirectoryReader.open(index)) {
                System.out.println("[i] Existing index found with " + reader.numDocs() + " chunk(s). Skipping re-index.");
                System.out.println("[i] Delete lucene-index/ and restart to force re-index.");
            }
        } else {
            System.out.println("[i] Building index from " + txtFiles.size() + " file(s)...");
            for (Path file : txtFiles) {
                indexFile(file.getFileName().toString(), Files.readString(file));
            }
            System.out.println("[i] Indexing complete.");
        }
    }


    // ── 7. REST API — GET / ───────────────────────────────────────────────────

    @Operation(summary = "Health check", description = "Confirms the API is running")
    @ApiResponse(responseCode = "200", description = "API is healthy")
    @GetMapping("/")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
            "status",  "ok",
            "message", "RAG API is running",
            "version", "1.0"
        ));
    }


    // ── 8. REST API — POST /documents ─────────────────────────────────────────

    @Operation(summary = "Upload and index a document",
               description = "Saves the file to data/ and indexes it immediately. No restart needed.")
    @ApiResponse(responseCode = "200", description = "File uploaded and indexed successfully")
    @ApiResponse(responseCode = "400", description = "No file provided")
    @PostMapping(value = "/documents", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<IndexResponse> uploadDocument(
        @Parameter(description = "The .txt file to upload", required = true)
        @RequestParam("file") MultipartFile file
    ) throws Exception {
        String filename = file.getOriginalFilename();
        if (filename == null || filename.isBlank()) {
            return ResponseEntity.badRequest().build();
        }
        Files.write(DATA_DIR.resolve(filename), file.getBytes());
        System.out.println("[+] Saved file: " + filename);
        int chunks = indexFile(filename, new String(file.getBytes()));
        return ResponseEntity.ok(new IndexResponse(filename, chunks));
    }


    // ── 9. REST API — POST /ask ───────────────────────────────────────────────

    @Operation(
        summary = "Ask a question",
        description = "Searches documents for context and asks Claude. Falls back to general knowledge if no documents match."
    )
    @ApiResponse(responseCode = "200", description = "Answer generated successfully")
    @ApiResponse(responseCode = "400", description = "Question is missing or blank")
    @io.swagger.v3.oas.annotations.parameters.RequestBody(content = @Content(examples = {
        @ExampleObject(name = "Simple question",       value = "{ \"question\": \"What is the capital of Golustaan?\" }"),
        @ExampleObject(name = "Continue conversation", value = "{ \"question\": \"Who rules it?\", \"conversationId\": \"abc-123\" }"),
        @ExampleObject(name = "Custom system prompt",  value = "{ \"question\": \"What is the capital?\", \"systemPrompt\": \"A capital is where parliament sits. Answer using the context below.\" }")
    }))
    @PostMapping("/ask")
    public ResponseEntity<AskResponse> ask(@RequestBody AskRequest request) throws Exception {
        if (request.question() == null || request.question().isBlank()) {
            return ResponseEntity.badRequest().build();
        }

        String conversationId = (request.conversationId() != null && !request.conversationId().isBlank())
            ? request.conversationId()
            : UUID.randomUUID().toString();

        List<Map<String, String>> history = loadConversation(conversationId);
        boolean hasCustomPrompt = request.systemPrompt() != null && !request.systemPrompt().isBlank();
        List<Map<String, String>> chunks = retrieve(request.question());

        String answer;
        String source;

        if (!chunks.isEmpty()) {
            StringBuilder context = new StringBuilder();
            for (int i = 0; i < chunks.size(); i++) {
                Map<String, String> chunk = chunks.get(i);
                context.append(String.format("[Chunk %d from '%s' (similarity: %s)]\n%s\n\n---\n\n",
                    i + 1, chunk.get("source"), chunk.get("score"), chunk.get("content")));
            }

            source = "📄 Answered from: " + chunks.stream()
                .map(c -> c.get("source")).distinct()
                .reduce((a, b) -> a + ", " + b).orElse("your documents");

            String basePrompt = hasCustomPrompt ? request.systemPrompt() : DEFAULT_RAG_PROMPT;
            answer = callClaudeApi(basePrompt + "\n\n=== CONTEXT ===\n" + context, history, request.question());

        } else {
            source = "🌐 Answered from: Claude's general knowledge (no relevant documents found)";
            answer = callClaudeApi(
                hasCustomPrompt ? request.systemPrompt() : DEFAULT_FALLBACK_PROMPT,
                history, request.question());
        }

        history.add(Map.of("role", "user",      "content", request.question()));
        history.add(Map.of("role", "assistant", "content", answer));
        saveConversation(conversationId, history);

        return ResponseEntity.ok(new AskResponse(conversationId, answer, source));
    }


    // ── 10. REST API — GET /conversations/{id} ────────────────────────────────

    @Operation(summary = "Get a specific conversation", description = "Returns full message history by ID")
    @ApiResponse(responseCode = "200", description = "Conversation found")
    @ApiResponse(responseCode = "404", description = "Conversation not found")
    @GetMapping("/conversations/{conversationId}")
    public ResponseEntity<Map<String, Object>> getConversation(
        @Parameter(description = "The conversation ID", example = "abc-123")
        @PathVariable String conversationId
    ) throws Exception {
        Path file = CONVERSATIONS_DIR.resolve(conversationId + ".json");
        if (!Files.exists(file)) return ResponseEntity.notFound().build();
        return ResponseEntity.ok(json.readValue(file.toFile(), Map.class));
    }


    // ── 11. REST API — GET /conversations ─────────────────────────────────────

    @Operation(summary = "List all conversations", description = "Paginated, most recent first")
    @ApiResponse(responseCode = "200", description = "Page of conversations returned")
    @GetMapping("/conversations")
    public ResponseEntity<ConversationsPage> listConversations(
        @Parameter(description = "Page number (0-based)") @RequestParam(defaultValue = "0")  int page,
        @Parameter(description = "Page size")             @RequestParam(defaultValue = "10") int size
    ) throws Exception {

        List<Path> allFiles;
        try (Stream<Path> stream = Files.list(CONVERSATIONS_DIR)) {
            allFiles = stream
                .filter(p -> p.toString().endsWith(".json"))
                .sorted((a, b) -> {
                    try { return Files.getLastModifiedTime(b).compareTo(Files.getLastModifiedTime(a)); }
                    catch (IOException e) { return 0; }
                }).toList();
        }

        int total      = allFiles.size();
        int totalPages = (int) Math.ceil((double) total / size);
        int fromIndex  = page * size;
        int toIndex    = Math.min(fromIndex + size, total);

        if (total > 0 && fromIndex >= total) return ResponseEntity.badRequest().build();

        List<Map<String, Object>> pageConversations = new ArrayList<>();
        for (Path file : (total == 0 ? List.<Path>of() : allFiles.subList(fromIndex, toIndex))) {
            pageConversations.add(json.readValue(file.toFile(), Map.class));
        }

        return ResponseEntity.ok(new ConversationsPage(pageConversations, page, size, total, totalPages));
    }


    // ── 12. Indexing ──────────────────────────────────────────────────────────

    private int indexFile(String name, String text) throws Exception {
        List<String> chunks = chunkText(text, CHUNK_SIZE, OVERLAP);
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);

        try (IndexWriter writer = new IndexWriter(index, config)) {
            for (String chunk : chunks) {
                float[] embedding = embed(chunk, "document");
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_VECTOR, embedding, VectorSimilarityFunction.DOT_PRODUCT));
                doc.add(new StringField(FIELD_SOURCE,  name,  Field.Store.YES));
                doc.add(new StoredField(FIELD_CONTENT, chunk));
                writer.addDocument(doc);
            }
        }

        System.out.println("[+] Indexed '" + name + "' → " + chunks.size() + " chunk(s)");
        return chunks.size();
    }

    private List<String> chunkText(String text, int chunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            chunks.add(text.substring(i, Math.min(i + chunkSize, text.length())));
            i += chunkSize - overlap;
        }
        return chunks;
    }


    // ── 13. Retrieval ─────────────────────────────────────────────────────────

    private List<Map<String, String>> retrieve(String question) throws Exception {
        List<Map<String, String>> results = new ArrayList<>();
        if (!DirectoryReader.indexExists(index)) return results;

        float[] queryVector = embed(question, "query");

        try (DirectoryReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_VECTOR, queryVector, TOP_K);
            ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;

            for (ScoreDoc hit : hits) {
                if (hit.score >= RELEVANCE_THRESHOLD) {
                    Document doc = searcher.storedFields().document(hit.doc);
                    results.add(Map.of(
                        "source",  doc.get(FIELD_SOURCE),
                        "content", doc.get(FIELD_CONTENT),
                        "score",   String.format("%.4f", hit.score)
                    ));
                }
            }
        }
        return results;
    }


    // ── 14. Voyage AI embedding ───────────────────────────────────────────────

    private float[] embed(String text, String inputType) throws Exception {
        String requestBody = json.writeValueAsString(Map.of(
            "model", VOYAGE_MODEL, "input", List.of(text), "input_type", inputType));

        String responseBody = httpSender.post(
            "https://api.voyageai.com/v1/embeddings",
            requestBody,
            Map.of("Content-Type", "application/json", "Authorization", "Bearer " + VOYAGE_KEY)
        );

        JsonNode embeddingNode = json.readTree(responseBody).path("data").get(0).path("embedding");
        float[] vector = new float[embeddingNode.size()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) embeddingNode.get(i).asDouble();
        }
        return vector;
    }


    // ── 15. Claude API call ───────────────────────────────────────────────────

    private String callClaudeApi(String systemPrompt, List<Map<String, String>> history, String newQuestion) throws Exception {
        List<Map<String, String>> messages = new ArrayList<>(history);
        messages.add(Map.of("role", "user", "content", newQuestion));

        String requestBody = json.writeValueAsString(Map.of(
            "model", CLAUDE_MODEL, "max_tokens", 1024,
            "system", systemPrompt, "messages", messages));

        String responseBody = httpSender.post(
            "https://api.anthropic.com/v1/messages",
            requestBody,
            Map.of("Content-Type", "application/json",
                   "x-api-key", API_KEY,
                   "anthropic-version", "2023-06-01")
        );

        return json.readTree(responseBody).path("content").get(0).path("text").asText();
    }


    // ── 16. Conversation persistence ──────────────────────────────────────────

    private List<Map<String, String>> loadConversation(String conversationId) throws Exception {
        Path file = CONVERSATIONS_DIR.resolve(conversationId + ".json");
        if (!Files.exists(file)) return new ArrayList<>();

        JsonNode root = json.readTree(file.toFile());
        List<Map<String, String>> messages = new ArrayList<>();
        for (JsonNode msg : root.path("messages")) {
            messages.add(Map.of("role", msg.path("role").asText(), "content", msg.path("content").asText()));
        }
        return messages;
    }

    private void saveConversation(String conversationId, List<Map<String, String>> messages) throws Exception {
        Path file = CONVERSATIONS_DIR.resolve(conversationId + ".json");
        Map<String, Object> conversation = new LinkedHashMap<>();
        conversation.put("conversationId", conversationId);
        conversation.put("updatedAt",      Instant.now().toString());
        conversation.put("messages",       messages);
        json.writeValue(file.toFile(), conversation);
    }


    // ── 17. Entry point ───────────────────────────────────────────────────────

    public static void main(String[] args) {
        SpringApplication.run(RagApp.class, args);
    }
}