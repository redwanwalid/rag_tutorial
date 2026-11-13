# Presenting the RAG pipeline

## Opening explanation

“This application answers questions over a book. At indexing time I split the source
into passages, embed them, and store the vectors and source metadata in a Qdrant
collection. At query time I embed the question using the same model, retrieve the
nearest passages, and give those passages to a chat model as evidence. The result is
an answer with references back to the retrieved text.”

## Five-minute walkthrough

1. Run `python -m rag_pipeline ingest --preview`. Explain why the whole book is not sent
   with every question, and show overlap and source offsets.
2. Build using `python -m rag_pipeline ingest` before the interview. During the demo,
   `python -m rag_pipeline query --stats` shows the persisted collection and vector dimension.
3. Run `python -m rag_pipeline query "How does Alice meet the Mad Hatter?" --search-only`.
   Explain semantic search and how metadata travels with the vector.
4. Repeat with `--show-context`. Point to the numbered passages in the prompt and
   their corresponding citations in the answer. Inspect support rather than assuming it.
5. Ask an unrelated question or raise `--min-score`. Explain both the retrieval gate
   (no passages means no chat call) and the model's instruction to abstain.
6. Show `rag_pipeline/commands/evaluate.py` and explain how retrieval quality can be measured
   independently of answer quality. Report the measured numbers, not expectations.

## Design questions to prepare for

| Question | Explanation |
|---|---|
| Why Qdrant? | It provides vector collections, payload metadata, filtering, persistence, and a search API. Local mode keeps the demo self-contained. |
| Exact search or HNSW? | Default embedded mode scans vectors exactly. Optional server mode configures HNSW, with background indexing and planner-selected scans for small/filtered sets. It is not a production recall/latency benchmark. |
| Why a recursive splitter? | It tries paragraph/sentence/word boundaries before falling back to characters. Chunk size trades retrieval precision against context completeness. |
| Why overlap? | It reduces loss of context at passage boundaries, at the cost of extra embedding/storage and repeated retrieved text. |
| Why cosine similarity? | It compares vector direction. It is a relevance signal, not a confidence probability or proof that a passage answers the question. |
| Why store the embedding model? | Same vector dimensions do not guarantee compatible embedding spaces. Query/document embeddings must use the same model. |
| Is this fine-tuning? | No model weights change. The application supplies retrieved evidence in each request. |
| What happens on rebuild failure? | Embeddings and a new collection are prepared first. The active manifest switches after point upload and count verification. Server graph construction can still be running in the background. |
| Can two processes query it? | Local mode locks the storage directory. Server mode supports concurrent reads; serialize builds that share an active manifest. |
| Does RAG eliminate hallucinations? | No. Retrieval can miss evidence, and a model can misread context. Grounded prompts, source display, evaluation, and abstention reduce risk. |
| What is missing for production? | Larger evidence labels, retrieval/answer evaluation, access control, incremental indexing, monitoring, latency/cost budgets, context budgeting, and prompt-injection defenses beyond instructions. |

## Useful experiments

- Rebuild with chunk sizes of 500, 1000, and 1500, then run the same retrieval cases.
- Compare top-k 1, 3, and 5. More context may improve recall but also add distractions.
- Add a second document and demonstrate `--source` filtering.
- Compare related vs unrelated phrases using `python -m rag_pipeline compare`.
- Inspect a failed answer: was evidence absent from retrieval, or present but misused?
- Try `RAG_VECTOR_MODE=server` using `compose.yaml`; inspect `query --stats` for indexed vectors. See the backend section in `pipeline.md` for tiny-corpus thresholds and endpoint isolation.

The bundled three-question evidence-substring check is deliberately small. Phrase matches
can produce false positives and fail on equivalent wording. A stronger evaluation would
label exact acceptable evidence passages, add paraphrased and unanswerable questions,
and assess answer correctness, citation support, and abstention separately.
