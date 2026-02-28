from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str=Field(description="The setup of the joke")
    punchline: str=Field(description="The punchline to the joke")

from langchain_openai import ChatOpenAI

model=ChatOpenAI(model="gpt-4o-mini", temperature=0)
model=model.with_structured_output(Joke)

r=model.invoke("Tell me a joke about cats")

print(r)
