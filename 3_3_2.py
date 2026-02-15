from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 2つのプロンプト
physics_template = """You are a very smart physics professor. You are great at
answering questions about physics in a concise and easy-to-understand manner.
When you don't know the answer to a question, you admit that you don't know.
Here is a question:{query}"""

math_template = """You are a very good mathematician. You are great at answering
math questions. You are so good because you are able to break down hard
problems into their component parts, answer the component parts, and then
put them together to answer the broader question.
Here is a question:{query}"""

# プロンプトを埋め込む
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)
# 質問をプロンプトにルーティングする
@chain
def prompt_router(query):
    query_embedding = embeddings.embed_query(query)
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # should be physics one
    t=PromptTemplate.from_template(most_similar)

    print(t)
    return t

semantic_router = (
prompt_router
| ChatOpenAI()
| StrOutputParser()
)
print(semantic_router.invoke("What's a black hole"))
