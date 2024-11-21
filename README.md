# Langchain RAG Tutorial

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additional help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the environment variable path.


2. Now run this command to install dependencies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown dependencies with: 

```python
pip install "unstructured[md]"
```

4. Install NLTK tokenizers

```
python -m nltk.downloader tokenizers
```

5. Generate an [OpenAI](https://platform.openai.com/settings) API key and save it to the `.env` file

```
OPENAI_API_KEY=
```

For the API key to function correctly, you will need to put some money on your account. The minimum of 5$ will suffice.

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
