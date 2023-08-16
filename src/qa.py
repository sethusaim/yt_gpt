from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def ask_query(query: str, api_key: str, index_name: str) -> str:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        vectordb = FAISS.load_local(
            folder_path="embeddings", index_name=index_name, embeddings=embeddings
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
        )

        res = qa_chain.run(query)

        return res

    except Exception as e:
        raise e
