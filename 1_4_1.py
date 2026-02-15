from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import pprint

class AnswerWithJustification(BaseModel):
    '''ユーザーの質問への回答と、その回答の正当性（根拠）を含むデータモデル'''
    answer: str
    '''ユーザーの質問への回答'''
    justification: str
    '''回答の正当性を示す根拠'''

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# you can't pass `BaseModel`
structured_llm = llm.with_structured_output(AnswerWithJustification)

r=structured_llm.invoke("What weighs more, a pound of bricks or a pound of feathers")

# how the API fill answer and justificaion?
pprint.pp(r)
