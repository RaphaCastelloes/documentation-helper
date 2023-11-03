import time
from typing import Set
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


st.header("LangChain Udemy Course - Documentation Helper Bot")

prompt = st.text_input("Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state['user_prompt_history'] = []

if "chat_answer_history" not in st.session_state:
    st.session_state['chat_answer_history'] = []
    
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
    
def create_sources_string(sources_urls: Set[str]) -> str:
    if not sources_urls:
        return ''
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Thinking..."):
        generated_response = run_llm(question=prompt, chat_history=st.session_state['chat_history'])
        sources = [doc.metadata["source"]
                   for doc in generated_response["source_documents"]]

        formatted_respose = f"{generated_response['answer']} \n\n {create_sources_string(sources)}"

        st.session_state['user_prompt_history'].append(prompt)
        st.session_state['chat_answer_history'].append(formatted_respose)
        st.session_state['chat_history'].append((prompt, generated_response["answer"]))

        print(formatted_respose)
        
if st.session_state['chat_answer_history']:
    for user_query, generated_response in zip(st.session_state['user_prompt_history'], st.session_state['chat_answer_history']):
        message(user_query, is_user=True)
        message(generated_response)