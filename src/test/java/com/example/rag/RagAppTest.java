package com.example.rag;

/**
 * Test suite for RagApp — works on Java 17, 21, and 25.
 *
 * Strategy:
 *   RagApp now exposes an HttpSender interface for all outbound HTTP calls.
 *   Tests simply replace the httpSender field with a lambda that returns
 *   hardcoded JSON responses — no Mockito, no proxy, no subclassing needed.
 *
 * Run: mvn clean test
 */

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.lang.reflect.Field;
import java.nio.file.*;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest(properties = {
    "ANTHROPIC_API_KEY=test-anthropic-key",
    "VOYAGE_API_KEY=test-voyage-key"
})
@AutoConfigureMockMvc
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class RagAppTest {

    @Autowired private MockMvc mockMvc;
    @Autowired private RagApp  ragApp;
    @Autowired private ObjectMapper objectMapper;

    // Shared across ordered tests
    private static String sharedConversationId;

    // ── Fake HTTP responses ───────────────────────────────────────────────────

    // 1024-dim float vector — matches voyage-3 embedding dimension
    private static final String FAKE_VOYAGE_BODY;
    static {
        float[] vec = new float[1024];
        Arrays.fill(vec, 0.1f);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vec.length; i++) {
            sb.append(vec[i]);
            if (i < vec.length - 1) sb.append(",");
        }
        sb.append("]");
        FAKE_VOYAGE_BODY = "{\"data\":[{\"embedding\":" + sb + "}]}";
    }

    private static final String FAKE_CLAUDE_BODY =
        "{\"content\":[{\"type\":\"text\",\"text\":\"Mocked Claude answer.\"}]}";

    /**
     * Before each test: replace ragApp's httpSender with a lambda that
     * returns fake responses based on the URL — no real HTTP calls made.
     *
     * This works because HttpSender is a plain Java interface defined in RagApp.
     * Swapping it via reflection is safe and version-agnostic.
     */
    @BeforeEach
    void injectFakeHttpSender() throws Exception {
        RagApp.HttpSender fakeSender = (url, body, headers) -> {
            if (url.contains("voyageai.com")) return FAKE_VOYAGE_BODY;
            return FAKE_CLAUDE_BODY;
        };

        Field field = RagApp.class.getDeclaredField("httpSender");
        field.setAccessible(true);
        field.set(ragApp, fakeSender);
    }


    // =========================================================================
    // UNIT TESTS — invoke private methods directly, no HTTP involved
    // =========================================================================

    @Test @Order(1)
    @DisplayName("Unit: chunkText — short text produces one chunk")
    void chunkText_shortText_singleChunk() throws Exception {
        var m = RagApp.class.getDeclaredMethod("chunkText", String.class, int.class, int.class);
        m.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<String> chunks = (List<String>) m.invoke(ragApp, "Hello world", 400, 80);
        assertEquals(1, chunks.size());
        assertEquals("Hello world", chunks.get(0));
    }

    @Test @Order(2)
    @DisplayName("Unit: chunkText — long text produces multiple chunks")
    void chunkText_longText_multipleChunks() throws Exception {
        var m = RagApp.class.getDeclaredMethod("chunkText", String.class, int.class, int.class);
        m.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<String> chunks = (List<String>) m.invoke(ragApp, "A".repeat(500), 400, 80);
        assertTrue(chunks.size() >= 2);
    }

    @Test @Order(3)
    @DisplayName("Unit: chunkText — no chunk exceeds the max chunk size")
    void chunkText_respectsChunkSize() throws Exception {
        var m = RagApp.class.getDeclaredMethod("chunkText", String.class, int.class, int.class);
        m.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<String> chunks = (List<String>) m.invoke(ragApp, "B".repeat(1000), 400, 80);
        for (String chunk : chunks) {
            assertTrue(chunk.length() <= 400, "Chunk too long: " + chunk.length());
        }
    }

    @Test @Order(4)
    @DisplayName("Unit: chunkText — empty text produces no chunks")
    void chunkText_emptyText_noChunks() throws Exception {
        var m = RagApp.class.getDeclaredMethod("chunkText", String.class, int.class, int.class);
        m.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<String> chunks = (List<String>) m.invoke(ragApp, "", 400, 80);
        assertTrue(chunks.isEmpty());
    }

    @Test @Order(5)
    @DisplayName("Unit: loadConversation — returns empty list for unknown ID")
    void loadConversation_unknownId_returnsEmptyList() throws Exception {
        var m = RagApp.class.getDeclaredMethod("loadConversation", String.class);
        m.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<Map<String, String>> result =
            (List<Map<String, String>>) m.invoke(ragApp, "no-such-id-" + UUID.randomUUID());
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test @Order(6)
    @DisplayName("Unit: save + load conversation — round-trip preserves all messages")
    void saveAndLoadConversation_roundTrip() throws Exception {
        var save = RagApp.class.getDeclaredMethod("saveConversation", String.class, List.class);
        var load = RagApp.class.getDeclaredMethod("loadConversation", String.class);
        save.setAccessible(true);
        load.setAccessible(true);

        String id = "test-rt-" + UUID.randomUUID();
        List<Map<String, String>> messages = List.of(
            Map.of("role", "user",      "content", "What is Golustaan?"),
            Map.of("role", "assistant", "content", "A beautiful country.")
        );
        save.invoke(ragApp, id, messages);

        @SuppressWarnings("unchecked")
        List<Map<String, String>> loaded = (List<Map<String, String>>) load.invoke(ragApp, id);

        assertEquals(2, loaded.size());
        assertEquals("user",               loaded.get(0).get("role"));
        assertEquals("What is Golustaan?", loaded.get(0).get("content"));
        assertEquals("assistant",          loaded.get(1).get("role"));
        assertEquals("A beautiful country.", loaded.get(1).get("content"));

        Files.deleteIfExists(Path.of("conversations/" + id + ".json"));
    }

    @Test @Order(7)
    @DisplayName("Unit: saveConversation — overwrites existing file on update")
    void saveConversation_overwritesExistingFile() throws Exception {
        var save = RagApp.class.getDeclaredMethod("saveConversation", String.class, List.class);
        var load = RagApp.class.getDeclaredMethod("loadConversation", String.class);
        save.setAccessible(true);
        load.setAccessible(true);

        String id = "test-ow-" + UUID.randomUUID();
        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(Map.of("role", "user", "content", "First question"));
        save.invoke(ragApp, id, messages);

        messages.add(Map.of("role", "assistant", "content", "First answer"));
        save.invoke(ragApp, id, messages);

        @SuppressWarnings("unchecked")
        List<Map<String, String>> loaded = (List<Map<String, String>>) load.invoke(ragApp, id);
        assertEquals(2, loaded.size());

        Files.deleteIfExists(Path.of("conversations/" + id + ".json"));
    }


    // =========================================================================
    // INTEGRATION TESTS — test REST endpoints via MockMvc
    // =========================================================================

    @Test @Order(10)
    @DisplayName("Integration: GET / — health check returns 200")
    void healthCheck_returns200() throws Exception {
        mockMvc.perform(get("/"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("ok"))
            .andExpect(jsonPath("$.message").value("RAG API is running"));
    }

    @Test @Order(20)
    @DisplayName("Integration: POST /documents — uploads and indexes a file")
    void uploadDocument_validFile_returnsFilenameAndChunks() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file", "test-facts.txt", MediaType.TEXT_PLAIN_VALUE,
            "Golustaan is a beautiful country north of India. Its ruled by a King.".getBytes()
        );
        mockMvc.perform(multipart("/documents").file(file))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.filename").value("test-facts.txt"))
            .andExpect(jsonPath("$.chunks").isNumber());
    }

    @Test @Order(21)
    @DisplayName("Integration: POST /documents — returns 400 when no file provided")
    void uploadDocument_missingFile_returns400() throws Exception {
        mockMvc.perform(multipart("/documents"))
            .andExpect(status().isBadRequest());
    }

    @Test @Order(30)
    @DisplayName("Integration: POST /ask — returns answer and generates a conversationId")
    void ask_newQuestion_returnsAnswerAndConversationId() throws Exception {
        MvcResult result = mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{ \"question\": \"What is the capital of Golustaan?\" }"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.conversationId").isNotEmpty())
            .andExpect(jsonPath("$.answer").isNotEmpty())
            .andExpect(jsonPath("$.source").isNotEmpty())
            .andReturn();

        Map<?, ?> response = objectMapper.readValue(result.getResponse().getContentAsString(), Map.class);
        sharedConversationId = (String) response.get("conversationId");
    }

    @Test @Order(31)
    @DisplayName("Integration: POST /ask — continues existing session with conversationId")
    void ask_withConversationId_continuesSession() throws Exception {
        assertNotNull(sharedConversationId, "Depends on test 30");
        mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{ \"question\": \"Who rules it?\", \"conversationId\": \"" + sharedConversationId + "\" }"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.conversationId").value(sharedConversationId))
            .andExpect(jsonPath("$.answer").isNotEmpty());
    }

    @Test @Order(32)
    @DisplayName("Integration: POST /ask — accepts and uses a custom systemPrompt")
    void ask_withCustomSystemPrompt_returnsAnswer() throws Exception {
        mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{ \"question\": \"What is the capital?\", \"systemPrompt\": \"A capital is where parliament sits. Answer using the context below.\" }"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.answer").isNotEmpty());
    }

    @Test @Order(33)
    @DisplayName("Integration: POST /ask — returns 400 for blank question")
    void ask_blankQuestion_returns400() throws Exception {
        mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{ \"question\": \"\" }"))
            .andExpect(status().isBadRequest());
    }

    @Test @Order(34)
    @DisplayName("Integration: POST /ask — returns 400 for missing question field")
    void ask_missingQuestion_returns400() throws Exception {
        mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{}"))
            .andExpect(status().isBadRequest());
    }

    @Test @Order(35)
    @DisplayName("Integration: POST /ask — source field is always present in response")
    void ask_sourceFieldPresent() throws Exception {
        MvcResult result = mockMvc.perform(post("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{ \"question\": \"What is the boiling point of nitrogen?\" }"))
            .andExpect(status().isOk())
            .andReturn();

        Map<?, ?> response = objectMapper.readValue(result.getResponse().getContentAsString(), Map.class);
        String source = (String) response.get("source");
        assertNotNull(source);
        assertFalse(source.isBlank(), "Source should not be blank");
    }

    @Test @Order(40)
    @DisplayName("Integration: GET /conversations/{id} — returns 200 for known ID")
    void getConversation_knownId_returnsConversation() throws Exception {
        assertNotNull(sharedConversationId, "Depends on test 30");
        mockMvc.perform(get("/conversations/" + sharedConversationId))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.conversationId").value(sharedConversationId))
            .andExpect(jsonPath("$.messages").isArray())
            .andExpect(jsonPath("$.updatedAt").isNotEmpty());
    }

    @Test @Order(41)
    @DisplayName("Integration: GET /conversations/{id} — returns 404 for unknown ID")
    void getConversation_unknownId_returns404() throws Exception {
        mockMvc.perform(get("/conversations/this-id-does-not-exist"))
            .andExpect(status().isNotFound());
    }

    @Test @Order(50)
    @DisplayName("Integration: GET /conversations — returns paginated list with metadata")
    void listConversations_returnsPagedResult() throws Exception {
        mockMvc.perform(get("/conversations").param("page", "0").param("size", "10"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.conversations").isArray())
            .andExpect(jsonPath("$.page").value(0))
            .andExpect(jsonPath("$.size").value(10))
            .andExpect(jsonPath("$.total").isNumber())
            .andExpect(jsonPath("$.totalPages").isNumber());
    }

    @Test @Order(51)
    @DisplayName("Integration: GET /conversations — uses default pagination when params omitted")
    void listConversations_defaultPagination() throws Exception {
        mockMvc.perform(get("/conversations"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.page").value(0))
            .andExpect(jsonPath("$.size").value(10));
    }

    @Test @Order(52)
    @DisplayName("Integration: GET /conversations — most recent conversation appears first")
    void listConversations_mostRecentFirst() throws Exception {
        assertNotNull(sharedConversationId);
        MvcResult result = mockMvc.perform(get("/conversations").param("page", "0").param("size", "10"))
            .andExpect(status().isOk())
            .andReturn();

        Map<?, ?> response = objectMapper.readValue(result.getResponse().getContentAsString(), Map.class);
        List<?> conversations = (List<?>) response.get("conversations");

        if (conversations.size() >= 2) {
            String first  = (String) ((Map<?, ?>) conversations.get(0)).get("updatedAt");
            String second = (String) ((Map<?, ?>) conversations.get(1)).get("updatedAt");
            assertTrue(first.compareTo(second) >= 0,
                "First conversation should be more recent");
        }
    }
}