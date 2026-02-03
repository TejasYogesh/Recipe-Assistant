import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(collection_name="recipe_chats", embedding_function=embeddings)
# OPENAI_API_KEY=""

template = """
You are a cooking assistant. Use the recipe context and past chats to answer.
Context: {context}
Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

def answer_question(question: str, context: str = ""):
    # Store chat in VectorDB
    vectorstore.add_texts([f"Context: {context}\nQuestion: {question}"])

    # Retrieve similar chats
    docs = vectorstore.similarity_search(question, k=3)
    past_context = "\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(
        context=context + "\n" + past_context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response.content

    # Store chat in VectorDB
    vectorstore.add_texts([f"Context: {context}\nQuestion: {question}"])

    # Retrieve similar chats
    docs = vectorstore.similarity_search(question, k=3)
    past_context = "\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context + "\n" + past_context, question=question)
    response = llm.predict(final_prompt)
    return response