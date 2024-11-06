import asyncio
import os
import time
import pandas as pd
import streamlit as st
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities, read_indexer_reports, read_indexer_relationships,
    read_indexer_text_units
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from callback import StreamlitLLMCallback
from search import CustomSearch

# Input directory for indexed data
INPUT_DIR = "./indexing/output/20241106-210853/artifacts"

# Main UI setup
st.title("GraphRAG Chatbot POC")

# Sidebar configurations
search_mode = st.sidebar.selectbox(
    "Choose search mode",
    options=['local', 'global'],
    help="Method to use to answer the query, one of local or global."
)

response_type = st.sidebar.selectbox(
    "Choose response type",
    options=['Single Paragraph', 'Multiple Paragraphs', 'Single Sentence',
             'List of 3-7 Points', 'Single Page', 'Multi-Page Report'],
    help="Free-form text describing the desired response type and format"
)

community_level = st.sidebar.number_input(
    "Community level",
    value=2,
    help="Community level in the Leiden community hierarchy from which we will load the community reports. Higher value means we use reports on smaller communities."
)

# Database configuration constants
LANCEDB_URI = "lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"


@st.cache_data(show_spinner=False)
def load_data(input_dir, community_level):
    """
    Load and process all required data files from the input directory.
    Uses Streamlit caching to avoid reloading data unnecessarily.
    """
    # Load parquet files
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")

    # Process data through indexer adapters
    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    reports = read_indexer_reports(report_df, entity_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units = read_indexer_text_units(text_unit_df)

    return entities, reports, relationships, text_units


@st.cache_resource(show_spinner=False)
def setup_vector_store(input_dir, community_level):
    """
    Initialize and setup the vector store for entity descriptions.
    Uses Streamlit resource caching to maintain the vector store across reruns.
    """
    entities, _, _, _ = load_data(input_dir, community_level)
    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities,
        vectorstore=description_embedding_store
    )
    return description_embedding_store, entity_description_embeddings


# Load data and setup vector store with loading spinner
with st.spinner("Loading data and setting up vector store..."):
    entities, reports, relationships, text_units = load_data(INPUT_DIR, community_level)
    description_embedding_store, entity_description_embeddings = setup_vector_store(INPUT_DIR, community_level)

# Clear input directory changed flag if present
if 'input_dir_changed' in st.session_state:
    del st.session_state.input_dir_changed

# LLM and embedding model setup
api_key = os.environ.get("OPENAI_API_KEY")
llm_model = st.sidebar.selectbox(
    "Choose LLM model",
    options=["gpt-4o-mini"],
    index=0
)
embedding_model = "text-embedding-3-small"

# Initialize LLM based on model selection
if llm_model.startswith("@cf"):
    llm = ChatOpenAI(
        api_key="ai2024",
        api_base="https://openai-cf.mggis0or1.workers.dev/v1",
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )
else:
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

# Initialize token encoder and text embedder
token_encoder = tiktoken.get_encoding("cl100k_base")
text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_base=None,
    api_type=OpenaiApiType.OpenAI,
    model=embedding_model,
    deployment_name=embedding_model,
    max_retries=20,
)


def setup_global_search():
    """
    Configure and return a GlobalSearch instance for handling global search queries.
    """
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    return GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type=response_type
    )


def setup_local_search():
    """
    Configure and return a CustomSearch instance for handling local search queries.
    """
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 5,
        "top_k_relationships": 5,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    streamlit_callback = StreamlitLLMCallback()

    return CustomSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        callbacks=[streamlit_callback]
    )


def get_initial_message():
    """Return the initial greeting message based on selected language."""
    return "How can I help you?"


# Initialize or reset chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state["messages"] or st.sidebar.button("Clear message history"):
    initial_message = get_initial_message()
    st.session_state["messages"] = [{"role": "assistant", "content": initial_message}]
elif st.session_state["messages"][0]["role"] == "assistant":
    current_initial_message = get_initial_message()
    if st.session_state["messages"][0]["content"] != current_initial_message:
        st.session_state["messages"][0]["content"] = current_initial_message

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
user_query = st.chat_input(placeholder="Ask me anything")

if user_query:
    start_time = time.time()
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # Initialize appropriate search engine based on mode
        search_engine = setup_global_search() if search_mode == 'global' else setup_local_search()


        async def perform_search():
            """Execute the search asynchronously."""
            return await search_engine.asearch(user_query)


        # Perform search and display results
        with st.spinner("Searching for an answer..."):
            result = asyncio.run(perform_search())

        response = result.response
        st.session_state.messages.append({"role": "assistant", "content": response})

        if search_mode == "global":
            st.write(response)
            if 'reports' in result.context_data.keys():
                with st.expander("View Source Data"):
                    st.write(result.context_data["reports"])

        # Display source data if available
        if 'sources' in result.context_data.keys():
            with st.expander("View Source Data"):
                st.write(result.context_data['sources'])

        # Display performance metrics
        latency = "N/A" if not hasattr(result, 'latency') else round(result.latency, 2)
        st.write(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}, latency: {latency}s")