from langchain_core.prompts import ChatPromptTemplate

answer_with_context_prompt = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("human", "### Context\n{context}\n### Question\n{question}")
])