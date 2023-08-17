from typing import List

from langchain.document_loaders.blob_loaders import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def generate_embeddings(
    urls: List[str], data_save_dir: str, api_key: str, index_name: str
) -> None:
    try:
        loader = GenericLoader(
            YoutubeAudioLoader(urls, data_save_dir),
            OpenAIWhisperParser(api_key=api_key),
        )

        docs = loader.load()

        combined_docs = [doc.page_content for doc in docs]
        text = " ".join(combined_docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )

        splits = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        vectordb = FAISS.from_texts(splits, embeddings)

        vectordb.save_local(folder_path="embeddings", index_name=index_name)
        
    except Exception as e:
        raise e
