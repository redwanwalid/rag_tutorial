"""Demonstrate semantic similarity using the same embedding provider as RAG."""
import argparse
import math
from rag_pipeline.config import Settings
from rag_pipeline.providers import OpenAIProvider
from rag_pipeline.cli import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', nargs='?', default='apple')
    parser.add_argument('second', nargs='?', default='iphone')
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    provider = OpenAIProvider()
    try:
        a, b = provider.embed([args.first, args.second], settings.embedding_model)
        similarity = sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)))
        print(f'Model: {settings.embedding_model} | dimensions: {len(a)}')
        print(f'{args.first!r} ↔ {args.second!r}')
        print(f'Cosine similarity: {similarity:.4f} | distance: {1-similarity:.4f}')
    finally:
        provider.close()


if __name__ == '__main__':
    run(main)
