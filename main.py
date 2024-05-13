import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

with st.sidebar:
    llm_name = st.radio(
        "# Choose you favorite LLM:",
        options=[
            "llama3-70b-8192",
            "llama3-8b-8192",
            "Gemma-7b-It",
            "Mixtral-8x7b-32768",
        ],
        captions=[
            "Meta's biggest and most powerful LLM.",
            "The little brother of llama3 70b.",
            "Google's open source LLM.",
            "Mixture-of-Experts model by Mistral AI. ",
        ],
    )

print(f"llm_name: {llm_name}")

llm = ChatGroq(
    temperature=0.2,
    model_name=llm_name,
)

template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}

Response:"""

prompt = PromptTemplate.from_template(template)

if "memory" not in st.session_state:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.memory = memory
    print(f"memory created {memory.load_memory_variables({})}")
else:
    memory = st.session_state.memory
    print(f"memory found {memory.load_memory_variables({})}")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("human").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

if prompt := st.chat_input("Enter your prompt"):
    st.chat_message("human").write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    try:
        response = conversation.run(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
