import os
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def get_corrected_text(api_key, text, model='llama3-8b-8192'):
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model
    )

    system_prompt = '''
    You are a grammar and spelling correction assistant. Your task is to correct the given sentences and return only the corrected version. Do not provide explanations or additional text. Here is an example:

    Input: "This is a exampel sentence with speling errors."
    Output: "This is an example sentence with spelling errors."

    Please correct the following sentence:

    {human_input}
    '''

    conversational_memory_length = 5
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history",
                                            return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    corrected_text = conversation.predict(human_input=text)
    return corrected_text


def get_word_predictions(api_key, text, model='llama3-8b-8192'):
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model
    )

    system_prompt = '''
    You are a word prediction assistant. Your task is to predict the next word in a given sentence based on context. Provide three recommended words that fit the context. Here is an example:

    Input: "She was very"
    Output: "happy, excited, surprised"

    Please predict the next word(s) for the following sentence (just print the words with out any other sentence):

    {human_input}
    '''

    conversational_memory_length = 5
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history",
                                            return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    prediction = conversation.predict(human_input=text)
    return prediction


def main():
    st.title('Text Correction and Word Prediction with Groq')

    api_key = st.text_input('Enter your Groq API key:', type='password')

    if api_key:
        tab1, tab2 = st.tabs(["Text Correction", "Word Prediction"])

        with tab1:
            st.header('Text Correction')
            input_text = st.text_area('Enter text to correct:')
            if st.button('Correct Text'):
                if input_text:
                    corrected_text = get_corrected_text(api_key, input_text)
                    st.subheader('Corrected Text:')
                    st.write(corrected_text)
                else:
                    st.error('Please provide text to correct.')

        with tab2:
            st.header('Word Prediction with Correction')
            input_text = st.text_area('Enter text for correction and prediction:', key="predict")

            # Automatically correct the text
            if input_text:
                corrected_text = get_corrected_text(api_key, input_text)
                st.subheader('Corrected Text:')
                st.write(corrected_text)

                # Provide predictions based on corrected text
                predictions = get_word_predictions(api_key, corrected_text)
                st.subheader('Recommended Words:')
                st.write(predictions)

    else:
        st.warning('Please enter your Groq API key.')


if __name__ == "__main__":
    main()
