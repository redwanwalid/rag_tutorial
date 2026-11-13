"""Run the demo with python -m rag_pipeline <command>."""
import argparse
import sys
from importlib import import_module
from .cli import run

COMMANDS = {
    'ingest': 'Load, split, embed, and store documents in Qdrant.',
    'query': 'Retrieve passages and generate a cited answer.',
    'evaluate': 'Measure evidence retrieval with Hit@k and MRR@k.',
    'compare': 'Compare embeddings for two strings.',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='RAG interview demo: LangChain, OpenAI, and local or server Qdrant.')
    parser.add_argument('command', choices=COMMANDS, help='; '.join(f'{name}: {help_text}' for name, help_text in COMMANDS.items()))
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments[:1])
    remaining = arguments[1:]
    command = import_module(f'.commands.{args.command}', package='rag_pipeline')
    command.main(remaining)


if __name__ == '__main__':
    run(main)
