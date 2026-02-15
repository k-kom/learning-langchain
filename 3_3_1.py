from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
# データモデル
class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="""Given a user question, choose which
        most relevant for answering their question""")

# 関数呼び出しに対応したLLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
# プロンプト
system = """You are an expert at routing a user question to the appropriate data
source.
Based on the programming language the question is referring to, route it to the
relevant data source."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
# ルーターを定義する
router = prompt | structured_llm

question = """Why doesn't the following code work:
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result = router.invoke({"question": question})
