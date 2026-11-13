"""Evaluate evidence retrieval: Hit@k and MRR@k, without chat generation."""
import argparse
import json
from pathlib import Path
from rag_pipeline.config import ROOT, Settings
from rag_pipeline.pipeline import retrieve
from rag_pipeline.providers import OpenAIProvider
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.cli import run


def evaluate(cases, search, top_k):
    if not cases:
        raise ValueError('Evaluation cases cannot be empty.')
    hits = reciprocal_rank = 0.0
    for case in cases:
        if not case.get('question', '').strip() or not case.get('evidence', '').strip():
            raise ValueError('Each case needs nonempty question and evidence strings.')
        passages = search(case['question'], top_k)
        rank = next((i for i, passage in enumerate(passages, 1)
                     if case['evidence'].casefold() in passage['text'].casefold()), None)
        hits += rank is not None
        reciprocal_rank += 1 / rank if rank else 0
        print(f"{'PASS' if rank else 'MISS'} | rank={rank or '-'} | {case['question']}")
    return {'cases': len(cases), f'Hit@{top_k}': hits / len(cases),
            f'MRR@{top_k}': reciprocal_rank / len(cases)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'evaluation/questions.json')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error('--top-k must be positive.')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Cases must be a nonempty JSON list.')
    settings = Settings.from_env()
    with VectorStore.from_settings(settings) as store:
        store.manifest()
        provider = OpenAIProvider()
        try:
            metrics = evaluate(cases, lambda question, k: retrieve(question, provider, store, k), args.top_k)
            print(json.dumps(metrics, indent=2))
        finally:
            provider.close()


if __name__ == '__main__':
    run(main)
