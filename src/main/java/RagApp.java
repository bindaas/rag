/**
 * RAG (Retrieval-Augmented Generation) Application
 * Uses Apache Lucene (on-disk) with Voyage AI embeddings + Anthropic Claude for generation.
 *
 * Features:
 *   - Voyage AI (voyage-3) generates embeddings for chunks and queries
 *   - Lucene KnnFloatVectorField stores and searches embeddings (HNSW algorithm)
 *   - On-disk index persists between runs
 *   - Auto-loads all .txt files from data/ folder at startup
 *   - Interactive question loop (Ctrl-C to exit)
 *   - Falls back to Claude's general knowledge if no relevant chunks found
 *   - Clearly communicates which source answered the question
 *
 * Dependencies (pom.xml):
 *   - org.apache.lucene:lucene-core:9.10.0
 *   - org.apache.lucene:lucene-analysis-common:9.10.0
 *   - com.fasterxml.jackson.core:jackson-databind:2.17.0
 *
 * Run:
 *   mvn compile exec:java -Dexec.mainClass="RagApp" \
 *     -Dexec.jvmArgs="-Dapi.key=sk-ant-... -Dvoyage.key=pa-..."
 */

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Stream;

public class RagApp {

    // ── 1. Configuration ──────────────────────────────────────────────────────

    // Anthropic API key: for Claude generation
    private static final String API_KEY     = System.getProperty("api.key",     System.getenv("ANTHROPIC_API_KEY"));

    // Voyage AI API key: for generating embeddings
    private static final String VOYAGE_KEY  = System.getProperty("voyage.key",  System.getenv("VOYAGE_API_KEY"));

    private static final String CLAUDE_MODEL  = "claude-sonnet-4-20250514";

    // voyage-3 produces 1024-dimensional vectors
    // If you switch to voyage-3-lite use 512, voyage-3-large use 2048
    private static final String VOYAGE_MODEL  = "voyage-3";
    private static final int    EMBEDDING_DIM = 1024;

    // Cosine similarity threshold: 0.0 = anything goes, 1.0 = perfect match
    // voyage-3 embeddings use DOT_PRODUCT on normalized vectors = cosine similarity
    // Scores below this are considered "not relevant enough" → fallback to Claude
    private static final float RELEVANCE_THRESHOLD = 0.7f;

    // How many top chunks to retrieve
    private static final int TOP_K = 3;

    // Chunk settings
    private static final int CHUNK_SIZE = 400;  // characters per chunk
    private static final int OVERLAP    = 80;   // overlap between chunks

    // Lucene field names
    private static final String FIELD_VECTOR  = "vector";   // stores the embedding
    private static final String FIELD_SOURCE  = "source";   // stores the filename
    private static final String FIELD_CONTENT = "content";  // stores the raw chunk text

    // Folder paths
    private static final Path DATA_DIR  = Path.of("data");
    private static final Path INDEX_DIR = Path.of("lucene-index");

    // ── 2. Infrastructure ─────────────────────────────────────────────────────

    private final FSDirectory      index;
    private final StandardAnalyzer analyzer = new StandardAnalyzer();
    private final ObjectMapper     json     = new ObjectMapper();
    private final HttpClient       http     = HttpClient.newHttpClient();

    public RagApp() throws IOException {
        Files.createDirectories(INDEX_DIR);
        this.index = FSDirectory.open(INDEX_DIR);
    }


    // ── 3. Data folder loading ────────────────────────────────────────────────

    /**
     * Scan data/ and index every .txt file.
     * Skips re-indexing if the on-disk index already exists.
     * Delete lucene-index/ to force a full rebuild.
     */
    public void loadDataFolder() throws Exception {
        if (!Files.exists(DATA_DIR)) {
            Files.createDirectories(DATA_DIR);
            System.out.println("[!] Created empty data/ folder. Add .txt files there and restart.");
            return;
        }

        List<Path> txtFiles;
        try (Stream<Path> stream = Files.list(DATA_DIR)) {
            txtFiles = stream
                .filter(p -> p.toString().endsWith(".txt"))
                .sorted()
                .toList();
        }

        if (txtFiles.isEmpty()) {
            System.out.println("[!] No .txt files found in data/. Add some and restart.");
            return;
        }

        boolean indexExists = DirectoryReader.indexExists(index);

        if (indexExists) {
            try (DirectoryReader reader = DirectoryReader.open(index)) {
                System.out.println("[i] Found existing on-disk index with " + reader.numDocs() + " chunk(s). Skipping re-index.");
                System.out.println("[i] To re-index, delete the lucene-index/ folder and restart.\n");
            }
        } else {
            System.out.println("[i] Building index from " + txtFiles.size() + " file(s) in data/...");
            System.out.println("[i] Generating embeddings via Voyage AI — this may take a moment...\n");
            for (Path file : txtFiles) {
                addDocument(file.getFileName().toString(), Files.readString(file));
            }
            System.out.println("\n[i] Indexing complete.\n");
        }
    }


    // ── 4. Indexing ───────────────────────────────────────────────────────────

    /**
     * Split a document into chunks, embed each chunk via Voyage AI,
     * and store the embedding + raw text in Lucene.
     *
     * Lucene fields per chunk:
     *   FIELD_VECTOR  — KnnFloatVectorField: the 1024-dim embedding (searchable)
     *   FIELD_SOURCE  — StringField:         filename (stored, not analyzed)
     *   FIELD_CONTENT — StoredField:         raw chunk text (stored for retrieval)
     */
    public void addDocument(String name, String text) throws Exception {
        List<String> chunks = chunkText(text, CHUNK_SIZE, OVERLAP);

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        try (IndexWriter writer = new IndexWriter(index, config)) {
            for (int i = 0; i < chunks.size(); i++) {
                String chunk = chunks.get(i);

                // Call Voyage AI to embed this chunk
                float[] embedding = embed(chunk);

                Document doc = new Document();

                // KnnFloatVectorField: stores the vector and makes it searchable
                // DOT_PRODUCT on Voyage's normalized vectors = cosine similarity
                doc.add(new KnnFloatVectorField(FIELD_VECTOR, embedding, VectorSimilarityFunction.DOT_PRODUCT));

                // StringField: stored verbatim, not tokenized
                doc.add(new StringField(FIELD_SOURCE, name, Field.Store.YES));

                // StoredField: raw text stored so we can return it to Claude
                doc.add(new StoredField(FIELD_CONTENT, chunk));

                writer.addDocument(doc);
                System.out.print(".");  // progress indicator
            }
        }

        System.out.println();
        System.out.println("[+] Indexed '" + name + "' → " + chunks.size() + " chunk(s)");
    }

    /**
     * Split text into overlapping chunks.
     */
    private List<String> chunkText(String text, int chunkSize, int overlap) {
        List<String> chunks = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            chunks.add(text.substring(i, Math.min(i + chunkSize, text.length())));
            i += chunkSize - overlap;
        }
        return chunks;
    }


    // ── 5. Embedding via Voyage AI ────────────────────────────────────────────

    /**
     * Call the Voyage AI /v1/embeddings endpoint and return a float[] vector.
     *
     * Request body:
     * {
     *   "model": "voyage-3",
     *   "input": ["text to embed"],
     *   "input_type": "document"   ← use "query" when embedding the user's question
     * }
     *
     * Response:
     * {
     *   "data": [{ "embedding": [0.123, -0.456, ...] }]
     * }
     *
     * @param text      The text to embed
     * @param inputType "document" for chunks being indexed, "query" for user questions
     */
    private float[] embed(String text, String inputType) throws Exception {
        String requestBody = json.writeValueAsString(Map.of(
            "model",      VOYAGE_MODEL,
            "input",      List.of(text),
            "input_type", inputType       // helps Voyage optimize the embedding direction
        ));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.voyageai.com/v1/embeddings"))
            .header("Content-Type",  "application/json")
            .header("Authorization", "Bearer " + VOYAGE_KEY)
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("Voyage API error " + response.statusCode() + ": " + response.body());
        }

        // Parse the embedding array from the response
        JsonNode embeddingNode = json.readTree(response.body())
            .path("data").get(0)
            .path("embedding");

        float[] vector = new float[embeddingNode.size()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) embeddingNode.get(i).asDouble();
        }
        return vector;
    }

    // Convenience overload: default to "document" input type
    private float[] embed(String text) throws Exception {
        return embed(text, "document");
    }


    // ── 6. Retrieval ──────────────────────────────────────────────────────────

    /**
     * Embed the question and search Lucene for the nearest chunks using HNSW.
     *
     * KnnFloatVectorQuery finds the top-k chunks whose embeddings are closest
     * to the query embedding in vector space — this is semantic search.
     * Unlike BM25, it matches by meaning, not just shared keywords.
     */
    public List<Map<String, String>> retrieve(String question) throws Exception {
        List<Map<String, String>> results = new ArrayList<>();

        if (!DirectoryReader.indexExists(index)) {
            return results;
        }

        // Embed the question as a "query" (Voyage optimizes differently for queries vs docs)
        float[] queryVector = embed(question, "query");

        try (DirectoryReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // KnnFloatVectorQuery: vector similarity search using HNSW
            KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_VECTOR, queryVector, TOP_K);
            ScoreDoc[] hits = searcher.search(query, TOP_K).scoreDocs;

            System.out.println("[debug] Vector search returned " + hits.length + " hit(s)");

            for (ScoreDoc hit : hits) {
                System.out.println("[debug] hit score: " + String.format("%.4f", hit.score) +
                                   " (threshold: " + RELEVANCE_THRESHOLD + ")");

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


    // ── 7. Answer generation ──────────────────────────────────────────────────

    public record Answer(String text, String source) {}

    /**
     * Full RAG pipeline with fallback:
     *
     *   1. Embed the question via Voyage AI
     *   2. Search Lucene for semantically similar chunks
     *   3a. Chunks found  → RAG mode:     Claude answers from your documents
     *   3b. No chunks     → Fallback mode: Claude answers from general knowledge
     */
    public Answer ask(String question) throws Exception {

        List<Map<String, String>> chunks = retrieve(question);

        if (!chunks.isEmpty()) {
            // ── RAG mode ──────────────────────────────────────────────────────

            StringBuilder context = new StringBuilder();
            for (int i = 0; i < chunks.size(); i++) {
                Map<String, String> chunk = chunks.get(i);
                context.append(String.format(
                    "[Chunk %d from '%s' (similarity: %s)]\n%s\n\n---\n\n",
                    i + 1, chunk.get("source"), chunk.get("score"), chunk.get("content")
                ));
            }

            String sourceFiles = chunks.stream()
                .map(c -> c.get("source"))
                .distinct()
                .reduce((a, b) -> a + ", " + b)
                .orElse("your documents");

            String prompt = """
                You are a helpful assistant. Answer the user's question using the context below.
                Use reasonable inference when the answer is implied but not explicitly stated.
                If you are inferring rather than quoting directly, clearly indicate this.
                If the context contains no relevant information at all, say so.

                === CONTEXT ===
                %s
                === QUESTION ===
                %s
                """.formatted(context.toString(), question);

            return new Answer(callClaudeApi(prompt), "📄 Answered from: " + sourceFiles);

        } else {
            // ── Fallback mode ─────────────────────────────────────────────────

            String prompt = """
                You are a helpful assistant. Answer the following question using your general knowledge.

                Question: %s
                """.formatted(question);

            return new Answer(
                callClaudeApi(prompt),
                "🌐 Answered from: Claude's general knowledge (no relevant documents found)"
            );
        }
    }


    // ── 8. Claude API call ────────────────────────────────────────────────────

    private String callClaudeApi(String prompt) throws Exception {
        String requestBody = json.writeValueAsString(Map.of(
            "model",      CLAUDE_MODEL,
            "max_tokens", 1024,
            "messages",   List.of(Map.of("role", "user", "content", prompt))
        ));

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.anthropic.com/v1/messages"))
            .header("Content-Type",      "application/json")
            .header("x-api-key",         API_KEY)
            .header("anthropic-version", "2023-06-01")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("Claude API error " + response.statusCode() + ": " + response.body());
        }

        return json.readTree(response.body())
            .path("content").get(0)
            .path("text").asText();
    }


    // ── 9. Interactive loop ───────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        System.out.println("===========================================");
        System.out.println("  RAG Assistant — Lucene + Voyage AI");
        System.out.println("  Press Ctrl-C to exit");
        System.out.println("===========================================\n");

        // Validate both API keys are present before doing anything
        if (API_KEY == null || API_KEY.isBlank()) {
            System.err.println("[error] Anthropic API key missing. Pass -Dapi.key=sk-ant-...");
            System.exit(1);
        }
        if (VOYAGE_KEY == null || VOYAGE_KEY.isBlank()) {
            System.err.println("[error] Voyage API key missing. Pass -Dvoyage.key=pa-...");
            System.exit(1);
        }

        RagApp rag = new RagApp();
        rag.loadDataFolder();

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String question = scanner.nextLine().trim();
            if (question.isEmpty()) continue;

            try {
                Answer answer = rag.ask(question);
                System.out.println("\nAssistant: " + answer.text());
                System.out.println(answer.source());
            } catch (Exception e) {
                System.out.println("[Error] " + e.getMessage());
            }

            System.out.println();
        }
    }
}