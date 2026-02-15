from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt_hyde = ChatPromptTemplate.from_template(
    """Please write a passage to
    answer the question.\n Question: {question} \n Passage:""")
generate_doc = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

retrieval_chain = generate_doc | retriever


prompt = ChatPromptTemplate.from_template("""Answer the following question based
on this context: {context}
Question: {question}
""")
llm = ChatOpenAI(temperature=0)

@chain
def qa(input):
    # HyDEリトリーバルチェーンから関連ドキュメントを取得する
    docs = retrieval_chain.invoke(input)
    # プロンプトを整形する
    formatted = prompt.invoke({"context": docs, "question": input})
    # 回答を生成する
    answer = llm.invoke(formatted)
    return answer

qa.invoke("""Who are some key figures in the ancient greek history of philosophy?""")
