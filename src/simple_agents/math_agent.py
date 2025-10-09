import os
import re
from typing  import List, Dict, Union
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.prebuilt import create_react_agent


##########################
# 1. Define the LLM
##########################

# Initialize the Hugging Face model
model = HuggingFaceEndpoint(
    model="Qwen/Qwen3-4B-Instruct-2507",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    temperature=0,
    max_new_tokens=512,
)

llm = ChatHuggingFace(llm=model)

##########################
# 2. Define the Tools
##########################

@tool
def add(a: int, b: int) -> int:
    """
    Adds two numbers and returns the result.
    Parameters:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of the two numbers.
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """
    Subtracts the second number from the first and returns the result.
    Parameters:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The result of the subtraction.
    """
    return a - b


@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Adds a list of numbers and returns the result. 

    Parameters:
        numbers (List[float]): A list of numbers to be added.
        absolute (bool): If True, adds the absolute values of the numbers.
    Returns:
        float: The sum of the numbers.
    """
    if absolute:
        numbers = [abs(num) for num in numbers]
    return sum(numbers)


@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimal numbers from the input string.

    Parameters:
    - inputs (str): A string that may contain numeric values.

    Returns:
    - dict: A dictionary with the key "result". If numbers are found, the value is their sum (float). 
            If no numbers are found or an error occurs, the value is a corresponding message (str).

    Example Input:
    "Add 10, 20.5, and -3."

    Example Output:
    {"result": 27.5}
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result": total}
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}
    

@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.
    
    Args:
        text: A string containing numbers that should be extracted and summed.
        
    Returns:
        The sum of all numbers found in the input.
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result

@tool
def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input string.
    Parameters:
    - inputs (str): 
    string, it should contain numbers that can be extracted and summed.
    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.
    Example Input:
    "Add the numbers 10, 20, and 30."
    Example Output:
    {"result": 60}
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    # numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]
    
    result = sum(numbers)
    return {"result": result}


@tool
def subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string, negates the first number, and successively subtracts 
    the remaining numbers in the list.

    This function is designed to handle input in string format, where numbers are separated 
    by spaces, commas, or other delimiters. It parses the string, extracts valid numeric values, 
    and performs a step-by-step subtraction operation starting with the first number negated.

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string may include spaces, commas, or 
      other delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Input:
    "100, 20, 10"

    Example Output:
    {"result": -130}

    Notes:
    - Non-numeric characters in the input are ignored.
    - If the input string contains only one valid number, the result will be that number negated.
    - Handles a variety of delimiters (e.g., spaces, commas) but does not validate input formats 
      beyond extracting numeric values.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number negated
    result = -1 * numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}

@tool
def multiply_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates their product.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the product of the numbers.

    Example Input:
    "2, 3, 4"

    Example Output:
    {"result": 24}

    Notes:
    - If no numbers are found, the result defaults to 1 (neutral element for multiplication).
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    print(numbers)

    # If no numbers are found, return 1
    if not numbers:
        return {"result": 1}

    # Calculate the product of the numbers
    result = 1
    for num in numbers:
        result *= num
        print(num)

    return {"result": result}

@tool
def divide_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first number 
    by the subsequent numbers in sequence.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the quotient.

    Example Input:
    "100, 5, 2"

    Example Output:
    {"result": 10.0}

    Notes:
    - If no numbers are found, the result defaults to 0.
    - Division by zero will raise an error.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]


    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Calculate the result of dividing the first number by subsequent numbers
    result = numbers[0]
    for num in numbers[1:]:
        result /= num

    return {"result": result}


@tool
def new_subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and performs subtraction sequentially, starting with the first number.

    This function is designed to handle input in string format, where numbers may be separated by spaces, 
    commas, or other delimiters. It parses the input string, extracts numeric values, and calculates 
    the result by subtracting each subsequent number from the first. inputs[0]-inputs[1]-inputs[2]

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string can include spaces, commas, or other 
      delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Usage:
    - Input: "100, 20, 10"
    - Output: {"result": 70}

    Limitations:
    - The function does not handle cases where numbers are formatted with decimals or other non-integer representations.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number
    result = numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}


def calculate_power(input_text: str) -> dict:
    """
    Calculates the power of a number (x^y).

    Parameters:
    - input_text (str): A string like "2, 3", "2 3", "5^2", or "2 to the power of 3".

    Returns:
    - dict: {"result": <calculated value>} or an error message.
    """
    # Try to extract expressions like "5^2"
    match = re.search(r"(\d+(?:\.\d+)?)\s*\^+\s*(\d+(?:\.\d+)?)", input_text)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}

    # Try to extract expressions like "2 to the power of 3"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to\s+the\s+power\s+of)\s*(\d+(?:\.\d+)?)", input_text, re.IGNORECASE)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}

    # Fallback: assume two numbers separated by space or comma
    try:
        numbers = [float(num) for num in input_text.replace(",", " ").split()]
        if len(numbers) != 2:
            return {"result": "Invalid input. Please provide exactly two numbers."}
        base, exponent = numbers
        return {"result": base ** exponent}
    except ValueError:
        return {"result": "Invalid input format. Provide input like '2 3', '2^3', or '2 to the power of 3'."}


# Create a Wikipedia tool using the @tool decorator
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.
    
    Parameters:
    - query (str): The topic or question to search for on Wikipedia
    
    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


##########################
# 3. Define the Agents
##########################

def agent_with_init_method():
    """
    This method uses the legacy AgentExecutor with ZeroShotAgent.
    It is simpler to use but less flexible and powerful compared to the graph-based agent.
    """
    # Note:
    # ZeroShotAgent does not support multi-input tools.
    # This method will be soon deprecated.
    # `create_react_agent` from LangGraph provides a more flexible and powerful alternative 
    # for building AI agents. This function creates a graph-based agent that works with 
    # chat models and supports tool-calling functionality.

    agent = initialize_agent(
        tools=[sum_numbers_from_text],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    response = agent.invoke("What is the sum of 10 and 5?")
    return response["output"]


def agent_with_langgraph():
    """
    Unlike the legacy AgentExecutor, which used a fixed loop structure, create_react_agent 
    creates a graph with these key nodes:

    Agent Node: Calls the LLM with the message history
    Tools Node: Executes any tool calls from the LLM's response
    Continue/End Nodes: Manage the workflow based on whether tool calls are present
    The graph follows this process:

    User message enters the graph
    LLM generates a response, potentially with tool calls
    If tool calls exist, they're executed and their results are added to the message history
    The updated messages are sent back to the LLM
    This loop continues until the LLM responds without tool calls
    The final state with all messages is returned
    """
    agent = create_react_agent(
        model=llm,
        tools=[add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers, search_wikipedia],
        # Optional: Add a system message to guide the agent's behavior
        prompt="""
        You are a helpful mathematical assistant that can perform various operations. 
        Use the tools precisely and explain your reasoning clearly.
        """.strip(),
        debug=False,
    )
    return agent


############################
# 4. Test the Tools & Agents
############################

def test_agent(agent):
    test_cases = [
        {
            "query": "Subtract 100, 20, and 10.",
            "expected": {"result": 70},
            "description": "Testing subtraction tool with sequential subtraction."
        },
        {
            "query": "Multiply 2, 3, and 4.",
            "expected": {"result": 24},
            "description": "Testing multiplication tool for a list of numbers."
        },
        {
            "query": "Divide 100 by 5 and then by 2.",
            "expected": {"result": 10.0},
            "description": "Testing division tool with sequential division."
        },
        {
            "query": "Subtract 50 from 20.",
            "expected": {"result": -30},
            "description": "Testing subtraction tool with negative results."
        }
    ]   

    correct_tasks = []
    # Corrected test execution
    for index, test in enumerate(test_cases, start=1):
        query = test["query"]
        expected_result = test["expected"]["result"]  # Extract just the value
        
        print(f"\n--- Test Case {index}: {test['description']} ---")
        print(f"Query: {query}")
        
        # Properly format the input
        response = agent.invoke({"messages": [("human", query)]})
        
        # Find the tool message in the response
        tool_message = None
        for msg in response["messages"]:
            if hasattr(msg, 'name') and msg.name in [
                'add_numbers', 'new_subtract_numbers', 'multiply_numbers', 'divide_numbers'
                ]:
                tool_message = msg
                break
        
        if tool_message:
            # Parse the tool result from its content
            import json
            tool_result = json.loads(tool_message.content)["result"]
            print(f"Tool Result: {tool_result}")
            print(f"Expected Result: {expected_result}")
            
            if tool_result == expected_result:
                print(f"✅ Test Passed: {test['description']}")
                correct_tasks.append(test["description"])
            else:
                print(f"❌ Test Failed: {test['description']}")
        else:
            print("❌ No tool was called by the agent")

    print("\nCorrectly passed tests:", correct_tasks)


if __name__ == "__main__":
    # test the add tool
    # result = add.invoke(
    #    {"a": 10, "b": 5}
    # )
    # print(f"Addition Result: {result}")
    # print("-----")
    # print("Tool name:", add.name)
    # print("Tool description:", add.description)
    # print("Tool parameters:", add.args)
    # print("-----")

    # # test the absolute add tool
    # result = add_numbers_with_options.invoke(
    #    {"numbers": [-10, 5, -3.5], "absolute": True}
    # )
    # print(f"Absolute Addition Result: {result}")
    # print("-----")

    agent = agent_with_langgraph()
    # test_agent(agent)

    query = "What is the population of Canada? Multiply it by 0.75"

    response = agent.invoke({"messages": [("human", query)]})

    print("\nMessage sequence:")
    for i, msg in enumerate(response["messages"]):
        print(f"\n--- Message {i+1} ---")
        print(f"Type: {type(msg).__name__}")
        if hasattr(msg, 'content'):
            print(f"Content: {msg.content}")
        if hasattr(msg, 'name'):
            print(f"Name: {msg.name}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"Tool calls: {msg.tool_calls}")
