from typing import TypedDict


# Process and summarize documents
class DocumentProcessingState(TypedDict):
    file_path: str
    extracted_text: str
    summary: str
    analysis_results: dict


# Customer support agent state
class SupportAgentState(TypedDict):
    user_query: str
    retrieved_documents: list
    formulated_response: str
    feedback: str