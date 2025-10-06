# Agentic Design Patterns

A comprehensive hands-on guide to building intelligent systems with **LangChain**, **LangGraph**, **CrewAI**, and **Model Context Protocol (MCP)**. This repository demonstrates practical implementations of key agentic design patterns essential for modern AI systems.

## 🎯 Overview

This project explores fundamental design patterns for building autonomous AI agents that can reason, plan, collaborate, and interact with external systems. Each pattern is implemented with working code examples using industry-standard frameworks.

## 🚀 Key Features

- **8 Core Workflow Patterns**: From sequential processing to multi-agent collaboration
- **Vector Database Integration**: ChromaDB for semantic search and document retrieval
- **RAG Systems**: Complete Retrieval-Augmented Generation implementations
- **Memory Management**: Persistent and contextual memory systems using LangGraph
- **Tool Integration**: Function calling and external system interactions
- **MCP Implementation**: Model Context Protocol for standardized AI-to-system communication
- **Multi-Agent Systems**: Collaborative agents with specialized roles
- **Goal-Driven Development**: Iterative code generation with objective tracking
- **Advanced Search**: Multi-modal filtering and intelligent recommendation systems

## 📁 Project Structure

```
agents_experimental/
├── main.py                     # Entry point
├── pyproject.toml             # Project dependencies
├── data/
│   └── FoodDataSet.json       # Sample dataset for RAG demonstrations
├── src/
│   ├── workflows/             # Core agentic patterns
│   │   ├── 1_sequential.py    # Sequential workflow pattern
│   │   ├── 2_routing.py       # Request routing and delegation
│   │   ├── 3_parallelization.py # Parallel processing
│   │   ├── 4_reflection.py    # Self-reflection pattern
│   │   ├── 5_tool_calling.py  # Function calling integration
│   │   ├── 6_planning.py      # Planning and execution
│   │   ├── 7_multi_agent.py   # Multi-agent collaboration
│   │   └── 8_goal_setting.py  # Goal-driven iterative development
│   ├── vector_databases/      # Vector DB and RAG implementations
│   │   ├── shared_functions.py # Common utilities for vector operations
│   │   ├── enhanced_rag_chatbot.py # Full RAG chatbot with LLM
│   │   ├── interactive_search.py # Interactive food recommendation
│   │   ├── advanced_search.py  # Advanced filtering and search
│   │   ├── calorie_checker.py  # Calorie-based food recommendations
│   │   ├── system_comparison.py # Comparison of different approaches
│   │   └── rag-cheatsheet.md   # Comprehensive RAG guide
│   ├── langgraph/             # LangGraph implementations
│   │   ├── example_state.py   # State management examples
│   │   └── lab - LangGraph101 Building Stateful AI Workflows.ipynb
│   ├── mcp/                   # Model Context Protocol
│   │   ├── client_adk.py      # MCP client implementation
│   │   ├── client_langgraph.py # LangGraph MCP integration
│   │   ├── server.py          # MCP server implementation
│   │   └── README.md          # MCP documentation
│   └── memory/                # Memory management systems
│       ├── memory_langgraph.py # LangGraph memory implementation
│       └── memory_management.py # Core memory patterns
├── gap_6843.py               # Generated code example
└── README.md                 # This file
```

## 🛠 Core Design Patterns

### 1. Sequential Workflow
**File**: `src/workflows/1_sequential.py`
- Implements chained processing with LangChain LCEL
- Demonstrates information extraction → transformation pipeline
- Uses prompt templating for multi-step operations

### 2. Routing & Delegation
**File**: `src/workflows/2_routing.py`
- Smart request routing based on content analysis
- Simulates specialized sub-agents (booking, info, general)
- Uses `RunnableBranch` for conditional logic

### 3. Parallelization
**File**: `src/workflows/3_parallelization.py`
- Concurrent task execution with `RunnableParallel`
- Demonstrates map-reduce pattern
- Synthesis of parallel results into comprehensive output

### 4. Tool Calling
**File**: `src/workflows/5_tool_calling.py`
- Integration with external functions and APIs
- LangChain agent framework implementation
- Simulated tool ecosystem for real-world scenarios

### 5. Multi-Agent Collaboration
**File**: `src/workflows/7_multi_agent.py`
- CrewAI implementation with specialized agent roles
- Sequential and parallel agent coordination
- Research analyst + technical writer collaboration example

### 6. Goal-Driven Development
**File**: `src/workflows/8_goal_setting.py`
- Iterative code generation with quality assessment
- LLM-based goal evaluation and feedback loops
- Automated code refinement and file generation

## 🗄️ Vector Databases & RAG Systems

### Enhanced RAG Chatbot
**File**: `src/vector_databases/enhanced_rag_chatbot.py`
- Complete RAG pipeline with LLM integration
- ChromaDB for vector storage and similarity search
- Contextual food recommendations with nutritional data
- LLM-powered natural language responses

### Interactive Search System
**File**: `src/vector_databases/interactive_search.py`
- CLI-based food recommendation chatbot
- Real-time similarity search with calorie filtering
- Search history and user preference tracking
- Interactive command system with help menus

### Advanced Search & Filtering
**File**: `src/vector_databases/advanced_search.py`
- Multi-modal filtering (cuisine, calories, ingredients)
- Combined search strategies demonstration
- Performance comparison between search approaches
- Metadata-based result refinement

### System Architecture Comparison
**File**: `src/vector_databases/system_comparison.py`
- Side-by-side comparison of different RAG approaches
- Performance benchmarking and response time analysis
- Trade-offs between simplicity and functionality
- Implementation complexity evaluation

## 🧠 Memory & State Management

### LangGraph Memory
**File**: `src/memory/memory_langgraph.py`
- Persistent memory using `InMemoryStore`
- Vector search and content filtering
- User context and conversation history management

### State Definitions
**File**: `src/langgraph/example_state.py`
- TypedDict implementations for different use cases
- Document processing and customer support states
- Structured data flow patterns

## 🔗 Model Context Protocol (MCP)

The MCP implementation demonstrates standardized communication between AI models and external systems:

- **Client-Server Architecture**: Standardized information flow
- **Resource Management**: External tool and data source integration
- **Protocol Compliance**: MCP standard implementation

## 🔧 Installation & Setup

### Prerequisites
- Python 3.12+
- API keys for:
  - OpenAI (optional)
  - Google Gemini
  - Groq
  - Anthropic (optional)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Rachneet/agentic-design-patterns.git
cd agentic-design-patterns
```

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Environment setup**:
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## 🚀 Usage Examples

### Run Sequential Workflow
```bash
python src/workflows/1_sequential.py
```

### Test Multi-Agent Collaboration
```bash
python src/workflows/7_multi_agent.py
```

### Goal-Driven Code Generation
```bash
python src/workflows/8_goal_setting.py
```

### Memory Management Demo
```bash
python src/memory/memory_langgraph.py
```

### Vector Database & RAG Examples
```bash
# Enhanced RAG chatbot with LLM
python src/vector_databases/enhanced_rag_chatbot.py

# Interactive food search system
python src/vector_databases/interactive_search.py

# Advanced filtering demonstrations
python src/vector_databases/advanced_search.py

# Compare different search approaches
python src/vector_databases/system_comparison.py
```

## 📚 Key Dependencies

- **LangChain**: Core framework for LLM applications
- **LangGraph**: State management and workflow orchestration
- **CrewAI**: Multi-agent system framework
- **FastMCP**: Model Context Protocol implementation
- **ChromaDB**: Vector database for similarity search and RAG
- **HuggingFace Transformers**: Embedding models and LLM integration
- **Google Generative AI**: Gemini model integration
- **Various LLM Providers**: OpenAI, Groq, Anthropic support

## 🎓 Learning Objectives

By exploring this repository, you will learn:

1. **Fundamental Agentic Patterns**: Core design principles for autonomous AI systems
2. **Workflow Orchestration**: Building complex, multi-step AI workflows
3. **Agent Collaboration**: Designing systems where multiple AI agents work together
4. **Memory Management**: Implementing persistent and contextual memory
5. **Tool Integration**: Connecting AI agents to external systems and APIs
6. **State Management**: Handling complex application state in AI systems
7. **Protocol Implementation**: Working with standardized AI communication protocols
8. **Vector Databases**: Implementing similarity search and retrieval systems
9. **RAG Architecture**: Building Retrieval-Augmented Generation pipelines
10. **Search & Filtering**: Advanced query processing and result refinement
11. **Performance Optimization**: Comparing and optimizing different search approaches

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different LLM providers
- Implement additional design patterns
- Extend existing examples
- Add new tool integrations

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or discussions about agentic design patterns, please open an issue in the repository.

---

**Note**: This repository contains working implementations of theoretical concepts. Ensure you have appropriate API keys and rate limits configured before running examples.