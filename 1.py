from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

import pprint

model = ChatOpenAI()

# system_msg = SystemMessage(
#     '''You are a helpful assistant that responds to questions with three exxlamation marks.''')
# human_msg = HumanMessage("What is the capital of France?")

# prompt = [system_msg, human_msg]


template = PromptTemplate.from_template("""Answer the question based on the
context below. If the question cannot be answered using the information
provided, answer with "I don't know".
Context: {context}
Question: {question}
Answer: """)

prompt = template.invoke({
"context": """The most recent advancements in NLP are being driven by Large
Language Models (LLMs). These models outperform their smaller
counterparts and have become invaluable for developers who are creating
applications with NLP capabilities. Developers can tap into these
models through Hugging Face's `transformers` library, or by utilizing
OpenAI and Cohere's offerings through the `openai` and `cohere`
libraries, respectively.""",
"question": "Which model providers offer LLMs?"
})

r= model.invoke(prompt)
pprint.pp(r)
