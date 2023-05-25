from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain


# QA_PROMPT

template = """
Role: 
Your name is MuseumBot, you are an AI assistant striving to provide helpful customer service for the Stedelijk Museum in Amsterdam, you are polite.

Instructions:
Provide a conversational answer to the user question, answer only on the basis of the following context and the chat history. 
If the answer cannot be drawn from the context or the chat history, just say "Unfortunately I don't know the answer to this question. Please reformulate it or ask something different". Don't try to make up an answer.
If the question is not about the Museum, politely inform them that you are tuned to only answer questions about the Museum.

Question: 
{question}

Context: 
{context}

Answer:
"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


# CONDENSE_QUESTION_PROMPT

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question to be about the Stedelijk Museum in Amsterdam or something mentioned in the chat history.

Chat History:
{chat_history}

Follow Up Input: 
{question}

Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


# CREATE QA CHAIN

def get_chain(vectorstore):

    # Set up LLM
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # Set the condense question generator
    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT
    )

    # use the streaming LLM to create a question answering chain
    doc_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=QA_PROMPT
    )

    # Set up chain
    qa_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )
    
    return qa_chain
