# requires: pip install langchain-postgres

'''
# postgresql+psycopg://langchain:langchain@localhost:6024/langchain
# run docker pgvector container
docker run \
  --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
'''

import uuid

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# ドキュメントを読み込んでチャンクに分割する
raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# チャンクごとに埋め込みを生成してベクトルストアに保存する
embeddings_model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents,
                             embeddings_model,
                             connection=connection)
