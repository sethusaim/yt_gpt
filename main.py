from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.blob_loaders import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]

save_dir = "./data/videos"

loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser(api_key="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"))

docs = loader.load()

combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

splits = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings(openai_api_key="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")

vectordb = FAISS.from_texts(splits, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

query = "Why do we need to zero out the gradient before backprop at each step?"

qa_chain.run(query)