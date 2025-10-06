import os

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",  # or another model of your choice
    base_url="https://api.groq.com/openai/v1",
    api_key = os.getenv("GROQ_API_KEY"),
)

class WeatherSchema(BaseModel):
    condition: str = Field(description="Weather condition such as sunny, rainy, cloudy")
    temperature: int = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit such as fahrenheit or celsius")


class SpamSchema(BaseModel):
    classification: str = Field(description="Email classification: spam or not_spam")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(description="Reason for the classification")

def extract_weather_response(response):
    tool_output = response.tool_calls[0]["args"]
    return {
        "condition": tool_output["condition"],
        "temperature": tool_output["temperature"],
        "unit": tool_output["unit"]
    }

def extract_spam_response(response):
    tool_output = response.tool_calls[0]["args"]
    return {
        "classification": tool_output["classification"],
        "confidence": tool_output["confidence"],
        "reason": tool_output["reason"]
    }


# weather_llm = llm.bind_tools(tools=[WeatherSchema])
# response = weather_llm.invoke("It's sunny and 75 degrees.")

spam_llm = llm.bind_tools(tools=[SpamSchema])
response = spam_llm.invoke("I'm a Nigerian prince, you want to be rich")
output = extract_spam_response(response)
print(output)
