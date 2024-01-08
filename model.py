from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    

    welcome_message = """


![Mining Image](https://wallpapercave.com/uwp/uwp4135324.png)

🚀 **Explore the Depths of Indian Mining with MinerGuide!**

Whether you're curious about mining processes, industry regulations, or the latest happenings in Indian mining, MinerGuide is your trusted companion.

🌐 **Ask Me Anything!**
I'm here to provide you with accurate and relevant information. If I don't have the answer, I'll let you know.

🔍 **Let's Dive In!**
Type your mining-related query, and let's embark on a journey through the fascinating world of mining in India.
"""
    
    msg = cl.Message(content=welcome_message)
    await msg.send()
    
    cl.user_session.set("chain", chain)

    # Hide settings and readme options
    cl.user_session.set("enable_settings", False)
    cl.user_session.set("enable_readme", False)



@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    answer += "\n\nFOR MORE INFORMATION VISIT - https://mines.gov.in/webportal/home"

    await cl.Message(content=answer).send()


if __name__ == "__main__":
    import subprocess
    subprocess.run(["chainlit", "run", "model.py"])