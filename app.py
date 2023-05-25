import streamlit as st
from streamlit_chat import message
import pickle
import os
from utils.query import get_chain


# Open vectorstore with FAQ data
if os.path.exists("faq_vectorstore.pkl"):
    with open("faq_vectorstore.pkl", "rb") as f:
        faq_vectorstore = pickle.load(f)

# Get the chain (see utils/query.py)
chain = get_chain(faq_vectorstore)


# Streamlit App

# Session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

# Title
st.title("Museum Bot 🏛️🤖 ")

# Description
st.markdown(
        ''' 
        > :blue[**A Chatbot built with Langchain, OpenAI GPT-3.5 and Streamlit
        > that can answer questions about the Stedelijk Museum Amsterdam based on the museum FAQ list**]
        ''')

# Input bar

def get_text():
    input_text = st.text_input(
        "You:", 
        value="", 
        key="input",
        placeholder="Enter your question here",
        label_visibility="hidden"
        )
    return input_text

user_input = get_text()

# Generate output upon user input

if user_input:

    # generate output
    output = chain.run(
        chat_history=[],
        question=user_input,
    )

    st.session_state.past.append(user_input)
    print(st.session_state["past"])
    st.session_state.generated.append(output)
    print(st.session_state["generated"])


    if st.session_state["generated"]:

        # loop backwards in the generated list
        for i in range(len(st.session_state["generated"]) -1, -1, -1):
            # st.info(st.session_state["past"][i], icon="👾")
            # st.success(st.session_state["generated"][i], icon="🤖")
            # pass the output generated by the chatbot
            message(st.session_state["generated"][i], key=str(i))
            # print the past input from the user
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            
