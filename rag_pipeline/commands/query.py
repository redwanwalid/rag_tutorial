"""Embed a question, retrieve Qdrant passages, and generate a cited answer."""
import argparse
import time
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import retrieve
from rag_pipeline.providers import OpenAIProvider, context_from_hits
from rag_pipeline.routing import small_talk_reply
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.cli import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--min-score', type=float, default=0.0, help='Cosine threshold, not a confidence probability.')
    parser.add_argument('--source', help='Exact relative filename to filter, e.g. alice_in_wonderland.md')
    parser.add_argument('--search-only', action='store_true', help='Show retrieval without calling the chat model; embedding still uses OpenAI.')
    parser.add_argument('--show-context', action='store_true', help='Display the complete retrieved passages sent to the model.')
    parser.add_argument('--stats', action='store_true', help='Inspect the active collection without any API calls.')
    args = parser.parse_args(argv)
    if not args.stats and (not args.question or not args.question.strip()):
        parser.error('Provide a nonempty question, or use --stats.')
    if args.top_k < 1 or not -1 <= args.min_score <= 1:
        parser.error('Require --top-k >= 1 and --min-score between -1 and 1.')
    # Explicit retrieval/statistics requests retain their original behavior.
    if not args.stats and not args.search_only:
        reply = small_talk_reply(args.question)
        if reply is not None:
            print('[ROUTE SMALL TALK] Retrieval and OpenAI calls skipped.\n\n' + reply)
            return
    settings = Settings.from_env()
    with VectorStore.from_settings(settings) as store:
        if args.stats:
            for key, value in store.stats().items():
                print(f'{key}: {value}')
            return
        manifest = store.manifest()
        provider = OpenAIProvider()
        try:
            started = time.perf_counter()
            print(f"[1/3 EMBED QUESTION] {manifest['embedding_model']}")
            hits = retrieve(args.question, provider, store, args.top_k, args.min_score, args.source)
            print(f'[2/3 RETRIEVE] {len(hits)} passages from Qdrant in {time.perf_counter()-started:.2f}s (including query embedding)')
            if not hits:
                print('No passages meet the search criteria. No chat call made.')
                return
            for i, hit in enumerate(hits, 1):
                print(f"[{i}] cosine={hit['score']:.4f} | {hit['source']} @ character {hit['start_index']}")
            if args.search_only or args.show_context:
                print('\nRETRIEVED CONTEXT\n' + context_from_hits(hits))
            if args.search_only:
                return
            started = time.perf_counter()
            answer = provider.answer(args.question, hits, settings.chat_model)
            print(f'\n[3/3 GENERATE] {settings.chat_model} in {time.perf_counter()-started:.2f}s\n\n{answer}')
        finally:
            provider.close()


if __name__ == '__main__':
    run(main)
