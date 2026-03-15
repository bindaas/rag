# RAG App

A simple RAG application using Apache Lucene + Voyage AI embeddings + Claude API.

## Setup

1. Add `.txt` files to the `data/` folder
2. Export your API keys:
```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export VOYAGE_API_KEY=pa-...
```
3. Run:
```bash
   mvn compile exec:java -Dexec.mainClass="RagApp"
```

## Re-indexing
To re-index after adding new files to `data/`:
```bash
rm -rf lucene-index/
```
Then restart the app.