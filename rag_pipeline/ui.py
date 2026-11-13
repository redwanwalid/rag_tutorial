"""Streamlit views for inspecting the RAG pipeline."""
import time

import streamlit as st

from .config import Settings
from .documents import load_documents, split_documents
from .pipeline import ingest, retrieve
from .providers import OpenAIProvider, context_from_hits
from .routing import small_talk_reply
from .ui_runtime import action, runtime
from .vector_store import VectorStore


def show_hits(hits):
    for number, hit in enumerate(hits, 1):
        with st.expander(f"[{number}] {hit['source']} · cosine {hit['score']:.4f}"):
            st.caption(f"Character offset: {hit['start_index']}. Cosine similarity is not a confidence probability.")
            st.text(hit['text'])
    if hits:
        with st.expander('Full retrieved context (used for answer generation)'):
            st.text(context_from_hits(hits))


def render_sidebar(settings):
    st.header('Retrieval settings')
    top_k = st.slider('Number of passages', 1, 20, 3)
    min_score = st.slider('Minimum cosine similarity', -1.0, 1.0, 0.0, 0.05)
    source = st.text_input('Source filter (exact relative filename)', placeholder='alice_in_wonderland.md')
    search_only = st.checkbox('Retrieve only (skip answer generation)')
    st.caption('Document questions use paid embeddings; generated document answers also use the chat model. Simple greetings are free.')
    st.divider()
    st.write(f'Backend: **{settings.vector_mode}**')
    st.write(f'Embedding: `{settings.embedding_model}`')
    st.write(f'Answer: `{settings.chat_model}`')
    st.caption(f'Documents: {settings.data_dir}')
    st.caption('The active index determines the embedding model used for questions.')
    if st.button('Clear conversation'):
        st.session_state.messages = []

    return top_k, min_score, source, search_only


def render_index(settings):
    _, log = runtime()
    st.subheader('Prepare your documents')
    st.write('Load Markdown and text files from the document folder configured in .env.')
    preview_col, stats_col = st.columns(2)
    with preview_col:
        if st.button('Preview chunks (free)'):
            with action('PREVIEW'):
                documents = load_documents(settings.data_dir)
                chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
                log.info('LOAD %d documents | SPLIT %d chunks | size=%d overlap=%d',
                         len(documents), len(chunks), settings.chunk_size, settings.chunk_overlap)
                st.success(f'{len(documents)} documents → {len(chunks)} chunks')
                for chunk in chunks[:3]:
                    st.caption(f"{chunk.metadata['source']} @ {chunk.metadata['start_index']}")
                    st.text(chunk.page_content)
    with stats_col:
        if st.button('Inspect index (free)'):
            with action('STATS'):
                with VectorStore.from_settings(settings) as store:
                    stats = store.stats()
                log.info('STATS | %d stored points | model=%s', stats['stored_points'], stats['embedding_model'])
                st.json(stats)
    st.divider()
    st.caption('Building or rebuilding embeds all document chunks and incurs OpenAI usage charges.')
    if st.button('Build / rebuild index', type='primary'):
        with action('INGEST'), st.spinner('Embedding documents and building the index…'):
            provider = OpenAIProvider()
            try:
                with VectorStore.from_settings(settings) as store:
                    manifest = ingest(settings, provider, store, report=log.info)
                st.success(f"Index ready: {manifest['chunk_count']} chunks")
                st.json(manifest)
            finally:
                provider.close()


def render_chat(settings, top_k, min_score, source, search_only, question=None):
    _, log = runtime()
    st.caption('Build an index in Documents & index first, or use your existing index. Backend stages appear in your terminal.')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'hits' in message:
                show_hits(message['hits'])
    if question and question.strip():
        st.session_state.messages.append(dict(role='user', content=question))
        with st.chat_message('user'):
            st.markdown(question)
        reply = None if search_only else small_talk_reply(question)
        if reply is not None:
            log.info('ROUTE SMALL TALK | retrieval and OpenAI calls skipped')
            with st.chat_message('assistant'):
                st.markdown(reply)
            st.session_state.messages.append(dict(role='assistant', content=reply))
            return
        with st.chat_message('assistant'), action('QUERY'), st.spinner('Searching documents…'):
            log.info('ROUTE RAG | document retrieval')
            with VectorStore.from_settings(settings) as store:
                manifest = store.manifest()
                provider = OpenAIProvider()
                try:
                    started = time.perf_counter()
                    log.info('EMBED QUESTION | model=%s | top_k=%d min_score=%.2f source=%s',
                             manifest['embedding_model'], top_k, min_score, source or '(all)')
                    hits = retrieve(question, provider, store, top_k, min_score, source.strip() or None)
                    log.info('RETRIEVE | %d passages | %.2fs including embedding', len(hits), time.perf_counter() - started)
                    for i, hit in enumerate(hits, 1):
                        log.info('SOURCE [%d] %s @ %d | cosine=%.4f', i, hit['source'], hit['start_index'], hit['score'])
                    if not hits:
                        answer = 'No passages meet the search criteria. Try a lower threshold or remove the source filter.'
                        log.info('GENERATE SKIPPED | no matching passages')
                    elif search_only:
                        answer = f'Retrieved {len(hits)} passages. Answer generation is disabled.'
                        log.info('GENERATE SKIPPED | retrieval only')
                    else:
                        started = time.perf_counter()
                        log.info('GENERATE START | model=%s | %d context characters', settings.chat_model, len(context_from_hits(hits)))
                        answer = provider.answer(question, hits, settings.chat_model)
                        log.info('GENERATE DONE | %.2fs | %d answer characters', time.perf_counter() - started, len(answer))
                finally:
                    provider.close()
            st.markdown(answer)
            show_hits(hits)
            st.session_state.messages.append(dict(role='assistant', content=answer, hits=hits))


def main():
    st.set_page_config(page_title='RAG explorer', page_icon='📚', layout='wide')
    st.title('📚 RAG explorer')
    st.write('Ask about your documents and inspect the passages behind each answer.')
    try:
        settings = Settings.from_env()
    except (ValueError, OSError) as exc:
        st.error(f'Configuration error: {exc}')
        st.stop()

    with st.sidebar:
        options = render_sidebar(settings)
    chat_tab, index_tab = st.tabs(['Ask documents', 'Documents & index'])
    # Main-body inputs are pinned by Streamlit; inside a tab they render inline.
    # Pass submissions into the chat view without moving the input into its layout.
    question = st.chat_input('Ask documents: What does the White Rabbit carry?', key='question')
    with index_tab:
        render_index(settings)
    with chat_tab:
        render_chat(settings, *options, question=question)
