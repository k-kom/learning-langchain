from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
import pprint

# indexing (takes so long)
# raw_documents = TextLoader('./test.txt').load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
# chunk_overlap=200)
# documents = text_splitter.split_documents(raw_documents)

# model = OpenAIEmbeddings()
# connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
# db = PGVector.from_documents(documents, model, connection=connection)

# retrieving related documents
retriever = db.as_retriever()
q = """Who are the key figures in the ancient greek history of philosophy?"""
related_docs = retriever.invoke(q)

# generate result
prompt = ChatPromptTemplate.from_template("""Answer the question based only on
 the following context:
 {context}
 Question: {question}
""")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
chain = prompt | llm

# 実行する
a=chain.invoke({"context": related_docs,"question": q})

pprint.pp(related_docs)
pprint.pp(a)

'''
AIMessage(content='The key figures in the ancient Greek history of philosophy mentioned in the context are:\n\n1. Thales of Miletus\n2. Anaximander\n3. Heraclitus\n4. Parmenides\n5. Socrates\n6. Plato (noted as a student of Socrates)', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 62, 'prompt_tokens': 917, 'total_tokens': 979, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f4ae844694', 'id': 'chatcmpl-D9QCWoVO80ROsjB0elS6vMC40QX8z', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--766e7435-74a9-4018-9d33-eeb19d8fc7f8-0', usage_metadata={'input_tokens': 917, 'output_tokens': 62, 'total_tokens': 979, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
'''
