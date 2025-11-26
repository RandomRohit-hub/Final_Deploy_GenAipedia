from langchain.prompts import ChatPromptTemplate

system_prompt = """
You are an AI assistant. Use the context below to answer the question.
If the context does not contain the answer, just say: "I don't know."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


