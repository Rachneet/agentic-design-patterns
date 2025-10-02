import os
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")  # type: ignore
)

# 1. Extract information prompt
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text.\n\n{text_input}"
)

# 2. Transform to JSON prompt
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the technical specifications into a JSON object with the keys: 'CPU', \
    'Memory', 'Storage'.\n\nSpecifications:\n{specifications}"
)

# 3. Build the chain using langchain LCEL
extraction_chain = prompt_extract | llm | StrOutputParser()

# The full chain passes the output of the extraction chain into the 'specifications'
full_chain = (
    {"specifications": extraction_chain} 
    | prompt_transform 
    | llm 
    | StrOutputParser()
)

# 4. Run the chain
input_text = """
The new SuperFast Laptop comes with a powerful Intel i7 processor, 16GB of RAM, 
and a 512GB SSD. It is designed for high performance and efficiency.
"""

output = full_chain.invoke({"text_input": input_text})
print(output)
