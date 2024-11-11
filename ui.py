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

# Constants and configurations
INPUT_DIR = "./indexing/output/demo/artifacts"
BASE_DOCUMENTS_PATH = f"{INPUT_DIR}/create_base_documents.parquet"
LANCEDB_URI = "lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"


# Load base documents data
@st.cache_data(show_spinner=False)
def load_base_documents():
    try:
        df = pd.read_parquet(BASE_DOCUMENTS_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading base documents: {str(e)}")
        return None


# Load data function
@st.cache_data(show_spinner=False)
def load_data(input_dir, community_level):
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    reports = read_indexer_reports(report_df, entity_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units = read_indexer_text_units(text_unit_df)

    return entities, reports, relationships, text_units


# Set up vector store
@st.cache_resource(show_spinner=False)
def setup_vector_store(input_dir, community_level):
    entities, _, _, _ = load_data(input_dir, community_level)
    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities,
        vectorstore=description_embedding_store
    )
    return description_embedding_store, entity_description_embeddings


def setup_global_search(llm, token_encoder, reports, entities, response_type):
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


def setup_local_search(llm, token_encoder, reports, text_units, entities, relationships,
                       description_embedding_store, text_embedder):
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


# Main application
def main():
    st.title("GraphRAG Chatbot POC")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Chat Interface", "Data Overview"],
        help="Select page to display"
    )

    if page == "Chat Interface":
        # Chat interface sidebar options
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

        # Set up LLM and embeddings
        api_key = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        text_embedder = OpenAIEmbedding(
            api_key=api_key,
            api_base=None,
            api_type=OpenaiApiType.OpenAI,
            model="text-embedding-3-small",
            deployment_name="text-embedding-3-small",
            max_retries=20,
        )

        # Load data and setup vector store
        with st.spinner("Loading data and setting up vector store..."):
            entities, reports, relationships, text_units = load_data(INPUT_DIR, community_level)
            description_embedding_store, entity_description_embeddings = setup_vector_store(
                INPUT_DIR,
                community_level
            )

        # Initialize chat messages
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

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
                if search_mode == 'global':
                    search_engine = setup_global_search(
                        llm, token_encoder, reports, entities, response_type
                    )
                else:
                    search_engine = setup_local_search(
                        llm, token_encoder, reports, text_units, entities, relationships,
                        description_embedding_store, text_embedder
                    )

                async def perform_search():
                    result = await search_engine.asearch(user_query)
                    return result

                with st.spinner("Searching for an answer..."):
                    result = asyncio.run(perform_search())

                response = result.response
                st.session_state.messages.append({"role": "assistant", "content": response})
                if search_mode == "global":
                    st.write(response)
                    if 'reports' in result.context_data.keys():
                        with st.expander("View Source Data"):
                            st.write(result.context_data["reports"])

                # Display context data
                if 'sources' in result.context_data.keys():
                    with st.expander("View Source Data"):
                        st.write(result.context_data['sources'])

                # Display LLM calls and tokens
                latency = "N/A"
                if hasattr(result, 'latency'):
                    latency = round(result.latency, 2)
                st.write(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}, latency: {latency}s")

    elif page == "Data Overview":
        st.header("Base Documents Overview")

        # Load and display base documents data
        with st.spinner("Loading base documents..."):
            df = load_base_documents()

        if df is not None:
            # Data summary
            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_data = df.isnull().sum().sum()
                st.metric("Missing Values", missing_data)

            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info)

            # Data preview
            st.subheader("Data Preview")

            # Add column selector
            selected_columns = st.multiselect(
                "Select columns to display",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:5]  # Default to first 5 columns
            )

            # Add search/filter functionality
            search_term = st.text_input("Search in text columns", "")

            # Filter data based on search term
            if search_term:
                mask = df[selected_columns].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                filtered_df = df[mask]
            else:
                filtered_df = df

            # Display data with pagination
            page_size = st.number_input("Rows per page", min_value=5, max_value=100, value=10)
            page_number = st.number_input("Page", min_value=1, max_value=(len(filtered_df) // page_size) + 1, value=1)

            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size

            st.dataframe(
                filtered_df[selected_columns].iloc[start_idx:end_idx],
                use_container_width=True
            )

if __name__ == "__main__":
    main()