import os
from typing import List, Dict, Tuple, Any
import streamlit as st
import pandas as pd
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import (
    ChatVectorDBChain,
    QAWithSourcesChain,
    VectorDBQAWithSourcesChain,
)
from langchain.prompts.prompt import PromptTemplate

from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from transformers import AutoTokenizer, AutoModel

# Set up OpenAI API key
# This is solely for the purpose of semantic search part of langchain vector search.
# Completion is still purely done using ChatGLM model.
os.environ["OPENAI_API_KEY"] = ""


@st.cache_resource()
def get_chat_glm():
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b-int4", trust_remote_code=True
    )
    model = (
        AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
        .half()
        .cuda()
    )
    model = model.eval()
    return model, tokenizer


def chat_with_agent(user_input, temperature=0.2, max_tokens=800, chat_history=[]):
    model, tokenizer = get_chat_glm()
    response, updated_history = model.chat(
        tokenizer,
        user_input,
        history=chat_history,
        temperature=temperature,
        max_length=max_tokens,
    )
    return response, updated_history


# Langchian related features
def init_wiki_agent(
    index_dir,
    max_token=800,
    temperature=0.3,
):

    embeddings = OpenAIEmbeddings()
    if index_dir:
        vectorstore = FAISS.load_local(index_dir, embeddings=embeddings)
    else:
        raise ValueError("Need saved vector store location")
    system_template = """使用以下文段, 简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息". 不要试图编造答案。 答案请使用中文.
----------------
{context}
----------------
"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    # qa = ChatVectorDBChain.from_llm(llm=ChatOpenAI(temperature=temperature, max_tokens=max_token),
    #                                  vectorstore=vectorstore,
    #                                  qa_prompt=prompt)

    condese_propmt_template = """任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。(确保问题是完整的, 没有模糊的指代)
聊天记录：
{chat_history}
###

后续问题：{question}

改写后的独立, 完整的问题："""
    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)

    from chatglm_llm import ChatGLM_G

    qa = ChatVectorDBChain.from_llm(
        llm=ChatGLM_G(),
        vectorstore=vectorstore,
        qa_prompt=prompt,
        condense_question_prompt=new_question_prompt,
    )
    qa.return_source_documents = True
    qa.top_k_docs_for_context = 3
    return qa


def get_wiki_agent_answer(query, qa, chat_history=[]):
    result = qa({"question": query, "chat_history": chat_history})
    return result
