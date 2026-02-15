import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

connection=""
collection_name=""
embeddings_model=OpenAIEmbeddings()

loader = TextLoader("./foo,txt", endocing="utf-8")
docs=loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_documents(docs)

prompt_text="Summarize the following document{doc}"

prompt = ChatPromptTemplate.from_template(prompt_text)
llm=ChatOpenAI(temperature=0,model="gpt-4o-mini")
summarize_chain = {
    "doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

summaries=summarize_chain.batch(chunks, {"max_concurrency":5})

vectorstore=PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

store = InMemoryStore()
id_key="doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

summary_docs=[
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, chunks)))

sub_docs = retriever.vectorstore.similarity_search("chapter of philosophy", k=2)

retrieved_docs = retriever.invoke("chapter on philosophy")

# 🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔🤔
