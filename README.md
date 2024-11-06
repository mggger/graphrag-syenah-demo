# GraphRAG Indexing and UI Development

This repository provides a guide to indexing training data and developing a UI using Streamlit for the GraphRAG project. Follow the steps below to set up and run the project.

## Training Data Indexing

### Indexing Data
1. **Train Knowledge Graph using GraphRAG**

   1. Initialize the indexing process:
      ```sh
      python -m graphrag.index --init --root ./indexing
      ```

   2. Configure the OpenAI API key:
      - Edit the `inputing/env` configuration file to include your OpenAI API key.

   3. Copy the generated `input` folder from the first step into `indexing/input`.

   4. Perform data indexing:
      ```sh
      python -m graphrag.index --root ./indexing
      ```

## UI Development with Streamlit

To develop and run the UI using Streamlit, use the following command:

```sh
export OPENAI_API_KEY="sk-xx"
streamlit run ui.py
```