"""
Planning enables agents to break down complex goals into actionable, sequential steps.
It is essential for handling multi-step tasks, workflow automation, and navigating complex environments.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables from .env file for security
load_dotenv()

# Set the environment variable for your chosen LLM provider (e.g., OPENAI_API_KEY)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # type: ignore
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # type: ignore
# Optionally specify a model, otherwise the default will be used.
os.environ["GROQ_MODEL_NAME"] = "llama-3.3-70b-versatile"


# # 1. Explicitly define the language model for clarity
# llm = ChatOpenAI(
#     model="llama-3.3-70b-versatile",  # Specify the model name
#     base_url="https://api.groq.com/openai/v1",  # Specify the base URL for the API
#     api_key=os.getenv("GROQ_API_KEY"),  # type: ignore
# )

# 2. Define a clear and focused agent
planner_writer_agent = Agent(
    role='Article Planner and Writer',
    goal='Plan and then write a concise, engaging summary on a specified topic.',
    backstory=(
        'You are an expert technical writer and content strategist. '
        'Your strength lies in creating a clear, actionable plan before writing, '
        'ensuring the final summary is both informative and easy to digest.'
    ),
    verbose=True,
    allow_delegation=False,
    # llm=llm # Assign the specific LLM to the agent
)

# 3. Define a task with a more structured and specific expected output
topic = "The importance of Reinforcement Learning in AI"
high_level_task = Task(
   description=(
       f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
       f"2. Write the summary based on your plan, keeping it around 200 words."
   ),
   expected_output=(
       "A final report containing two distinct sections:\n\n"
       "### Plan\n"
       "- A bulleted list outlining the main points of the summary.\n\n"
       "### Summary\n"
       "- A concise and well-structured summary of the topic."
   ),
   agent=planner_writer_agent,
)

# Create the crew with a clear process
crew = Crew(
   agents=[planner_writer_agent],
   tasks=[high_level_task],
   process=Process.sequential,
)

# Execute the task
print("## Running the planning and writing task ##")
result = crew.kickoff()

print("\n\n---\n## Task Result ##\n---")
print(result)
