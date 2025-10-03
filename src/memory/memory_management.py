"""
Memory is a critical component for creating intelligent and natural-feeling 
conversational applications.

It allows an AI agent to remember information from past interactions, 
learn from feedback, and adapt to user preferences. 

Short-Term Memory: This is thread-scoped, meaning it tracks the ongoing 
conversation within a single session or thread. It provides immediate context, 
but a full history can challenge an LLM's context window, potentially leading 
to errors or poor performance. LangGraph manages short-term memory as part of 
the agent's state, which is persisted via a checkpointer, allowing a thread to 
be resumed at any time.

Long-Term Memory: This stores user-specific or application-level data across 
sessions and is shared between conversational threads. It is saved in custom 
"namespaces" and can be recalled at any time in any thread. LangGraph provides 
stores to save and recall long-term memories, enabling agents to retain 
knowledge indefinitely.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----- ChatMessageHistory -----
# This is a simple example of using LangChain's ChatMessageHistory to manage
# conversational memory manually. This can be integrated into more complex agent systems
# as needed.

from langchain_community.chat_message_histories import ChatMessageHistory

def example_chat_history():
    history = ChatMessageHistory()
    # Adding Messages to History
    history.add_user_message("What is the weather like today?")
    history.add_ai_message("The weather is sunny with a high of 75°F.")
    # Retrieving Messages
    messages = history.messages
    for message in messages:
        print(f"{message.type}: {message.content}")
    return history


# ----- ConversationBufferMemory -----
# For integrating memory directly into chains, ConversationBufferMemory is a 
# common choice. It holds a buffer of the conversation and makes it available 
# to your prompt.

from langchain.memory import ConversationBufferMemory

def example_conversation_buffer():
    memory = ConversationBufferMemory()
    # Simulate a conversation
    memory.save_context(
        {"input": "Hello, who are you?"},
        {"output": "I am an AI created by Rachneet. How can I assist you today?"}
    )
    memory.save_context(
        {"input": "Can you tell me a joke?"},
        {"output": "Why did the scarecrow win an award? Because he was outstanding in his field!"}
    )
    # Retrieve the conversation history
    context = memory.load_memory_variables({})
    print(context)
    return memory


from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def example_chain_with_memory():
    # 1. Define LLM and Prompt
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  # type: ignore
    template = """You are a helpful travel agent.

    Previous conversation:
    {history}

    New question: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)

    # 2. Configure Memory
    # The memory_key "history" matches the variable in the prompt
    memory = ConversationBufferMemory(memory_key="history")

    # 3. Build the Chain
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # 4. Run the Conversation
    response = conversation.predict(question="I want to book a flight.")
    print(response)
    response = conversation.predict(question="My name is Sam, by the way.")
    print(response)
    response = conversation.predict(question="What was my name again?")
    print(response)


def example_chat_memory_with_chat_prompt():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)  # type: ignore

    prompt = ChatPromptTemplate(
   messages=[
       SystemMessagePromptTemplate.from_template("You are a friendly assistant."),
       MessagesPlaceholder(variable_name="chat_history"),
       HumanMessagePromptTemplate.from_template("{question}")
       ]
    )

    # 2. Configure Memory
    # return_messages=True is essential for chat models
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 3. Build the Chain
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # 4. Run the Conversation
    response = conversation.predict(question="Hi, I'm Jane.")
    print(response)
    response = conversation.predict(question="Do you remember my name?")
    print(response)


if __name__ == "__main__":
    # print("=== ChatMessageHistory Example ===")
    # example_chat_history()
    # print("\n=== ConversationBufferMemory Example ===")
    # example_conversation_buffer()
    # print("\n=== Chain with Memory Example ===")
    # example_chain_with_memory()
    print("\n=== Chat Memory with Chat Prompt Example ===")
    example_chat_memory_with_chat_prompt()



