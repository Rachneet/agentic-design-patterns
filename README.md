# Agentic Design Patterns & Parallelism

A comprehensive hands-on guide to building intelligent systems with **LangChain**, **LangGraph**, **CrewAI**, and **Model Context Protocol (MCP)**. This repository demonstrates practical implementations of agentic design patterns and advanced parallelism techniques for high-performance AI systems.

## 🎯 Overview

This project explores fundamental design patterns for building autonomous AI agents that can reason, plan, collaborate, and execute tasks in parallel. From basic sequential workflows to sophisticated parallel agent ensembles and advanced RAG architectures, each pattern is implemented with working code examples using industry-standard frameworks.

**What's New**: Extensive collection of **14 parallel agent patterns** including competitive ensembles, multi-hop retrieval, hybrid search fusion, sharded retrieval, and parallel context preprocessing. These patterns demonstrate how to build production-grade, high-performance AI systems that scale.

## 🚀 Key Features

### Core Patterns
- **8 Foundational Workflow Patterns**: Sequential processing, routing, parallelization, reflection, tool calling, planning, multi-agent collaboration, and goal-driven development
- **14 Advanced Parallelism Patterns**: Production-ready patterns for high-performance agent systems

### Parallelism & Performance
- **Parallel Tool Execution**: Concurrent tool calls for reduced latency
- **Competitive Agent Ensembles**: Multiple diverse agents competing for best outputs
- **Hierarchical Agent Teams**: Coordinated multi-level agent architectures
- **Speculative Execution**: Parallel hypothesis generation and validation
- **Redundant Execution**: Fault-tolerant parallel processing with voting mechanisms

### Advanced RAG Systems
- **Multi-Hop Retrieval**: Complex query decomposition with parallel sub-question answering
- **Hybrid Search Fusion**: Combining semantic and lexical search in parallel
- **Sharded Retrieval**: Distributed document retrieval across multiple indices
- **Parallel Context Preprocessing**: Concurrent document filtering and relevance checking
- **Query Expansion**: Parallel query reformulation strategies
- **LlamaIndex & LangChain Integration**: Comprehensive retriever implementations

### Infrastructure & Tools
- **Vector Database Integration**: ChromaDB, FAISS for high-performance similarity search
- **Memory Management**: Persistent and contextual memory with LangGraph
- **MCP Implementation**: Model Context Protocol for standardized AI communication
- **Knowledge Graphs**: Graph-based retrieval and reasoning
- **Interactive Notebooks**: 35+ hands-on examples with detailed explanations
- **Performance Benchmarking**: Comparative analysis and optimization strategies

## 📁 Project Structure

```
agents_experimental/
├── main.py                     # Entry point
├── pyproject.toml             # Project dependencies
├── data/
│   └── FoodDataSet.json       # Sample dataset for RAG demonstrations
├── src/
│   ├── workflows/             # Core agentic patterns (8 patterns)
│   │   ├── 1_sequential.py    # Sequential workflow pattern
│   │   ├── 2_routing.py       # Request routing and delegation
│   │   ├── 3_parallelization.py # Parallel processing
│   │   ├── 4_reflection.py    # Self-reflection pattern
│   │   ├── 5_tool_calling.py  # Function calling integration
│   │   ├── 5_1_tool_calling_crewai.py # CrewAI tool integration
│   │   ├── 6_planning.py      # Planning and execution
│   │   ├── 7_multi_agent.py   # Multi-agent collaboration
│   │   └── 8_goal_setting.py  # Goal-driven iterative development
│   │
│   ├── agents_parallelism/    # Advanced parallelism patterns (14 notebooks)
│   │   ├── 1_parallel_tool_use.ipynb           # Concurrent tool execution
│   │   ├── 2_parallel_hypothesis.ipynb         # Parallel hypothesis generation
│   │   ├── 3_parallel_evaluation.ipynb         # Concurrent evaluation strategies
│   │   ├── 4_speculative_execution.ipynb       # Speculative parallel execution
│   │   ├── 5_hierarchical_agent_teams.ipynb    # Multi-level agent coordination
│   │   ├── 6_competitive_agent_ensemble.ipynb  # Competitive agent voting
│   │   ├── 7_agent_assembly_line.ipynb         # Pipeline-based parallel processing
│   │   ├── 8_decentralized_blackboard.ipynb    # Shared state parallel agents
│   │   ├── 9_redundant_execution.ipynb         # Fault-tolerant parallel execution
│   │   ├── 10_parallel_query_expansion.ipynb   # Parallel query reformulation
│   │   ├── 11_sharded_retrieval.ipynb          # Distributed document retrieval
│   │   ├── 12_hybrid_search_fusion.ipynb       # Semantic + lexical search fusion
│   │   ├── 13_parallel_context_preprocessing.ipynb # Concurrent context filtering
│   │   └── 14_multi_hop_retrieval.ipynb        # Complex multi-step retrieval
│   │
│   ├── vector_databases/      # Vector DB and RAG implementations
│   │   ├── retrievers/         # Advanced retriever patterns
│   │   │   ├── langchain_retrievers.ipynb  # LangChain retrieval strategies
│   │   │   └── llamaindex_retrievers.ipynb # LlamaIndex retrieval patterns
│   │   ├── rag/                # RAG implementations
│   │   │   ├── rag_with_hf.ipynb    # HuggingFace-based RAG
│   │   │   └── rag_with_torch.ipynb # PyTorch-based RAG
│   │   ├── semantic_similarity/ # Similarity search techniques
│   │   │   └── Semantic Similarity with FAISS.ipynb
│   │   └── food_recommender_rag/ # Complete RAG system example
│   │       ├── shared_functions.py
│   │       ├── enhanced_rag_chatbot.py
│   │       ├── interactive_search.py
│   │       ├── advanced_search.py
│   │       └── rag-cheatsheet.md
│   │
│   ├── knowledge_graphs/      # Graph-based retrieval
│   │   ├── construct_kg.ipynb  # Knowledge graph construction
│   │   ├── query_kg.ipynb      # Graph querying patterns
│   │   ├── chat_with_kg.ipynb  # Conversational graph interfaces
│   │   └── text_for_rag.ipynb  # Text-to-graph for RAG
│   │
│   ├── langgraph/             # LangGraph implementations
│   │   ├── langgraph_components.ipynb # Core LangGraph building blocks
│   │   ├── agentic_search.ipynb       # Agent-based search patterns
│   │   ├── persistence_and_streaming.ipynb # State persistence
│   │   ├── human_in_loop.ipynb        # Human-in-the-loop workflows
│   │   ├── essay_writer_agent.ipynb   # Long-form content generation
│   │   ├── ReAct.ipynb                # ReAct pattern implementation
│   │   └── langgraph_cheatsheet.md    # LangGraph reference guide
│   │
│   ├── simple_agents/         # Beginner-friendly agent examples
│   │   ├── Build Interactive LLM Agents with Tools.ipynb
│   │   ├── Manual Tool-Calling Agent.ipynb
│   │   └── LLM-Powered Data Science LCEL.ipynb
│   │
│   ├── multimodal/            # Multimodal AI implementations
│   │   └── Build an Image Captioning System.ipynb
│   │
│   ├── mcp/                   # Model Context Protocol
│   │   ├── client_adk.py      # MCP client implementation
│   │   ├── client_langgraph.py # LangGraph MCP integration
│   │   ├── server.py          # MCP server implementation
│   │   └── README.md          # MCP documentation
│   │
│   └── memory/                # Memory management systems
│       ├── memory_langgraph.py # LangGraph memory implementation
│       └── memory_management.py # Core memory patterns
│
└── README.md                 # This file
```

## 🛠 Core Design Patterns

### 1. Sequential Workflow
**File**: [src/workflows/1_sequential.py](src/workflows/1_sequential.py)
- Implements chained processing with LangChain LCEL
- Demonstrates information extraction → transformation pipeline
- Uses prompt templating for multi-step operations

### 2. Routing & Delegation
**File**: [src/workflows/2_routing.py](src/workflows/2_routing.py)
- Smart request routing based on content analysis
- Simulates specialized sub-agents (booking, info, general)
- Uses `RunnableBranch` for conditional logic

### 3. Parallelization
**File**: [src/workflows/3_parallelization.py](src/workflows/3_parallelization.py)
- Concurrent task execution with `RunnableParallel`
- Demonstrates map-reduce pattern
- Synthesis of parallel results into comprehensive output

### 4. Tool Calling
**File**: [src/workflows/5_tool_calling.py](src/workflows/5_tool_calling.py)
- Integration with external functions and APIs
- LangChain agent framework implementation
- Simulated tool ecosystem for real-world scenarios

### 5. Multi-Agent Collaboration
**File**: [src/workflows/7_multi_agent.py](src/workflows/7_multi_agent.py)
- CrewAI implementation with specialized agent roles
- Sequential and parallel agent coordination
- Research analyst + technical writer collaboration example

### 6. Goal-Driven Development
**File**: [src/workflows/8_goal_setting.py](src/workflows/8_goal_setting.py)
- Iterative code generation with quality assessment
- LLM-based goal evaluation and feedback loops
- Automated code refinement and file generation

## ⚡ Advanced Parallelism Patterns

This section contains 14 production-ready patterns for building high-performance, scalable AI systems. Each pattern addresses specific performance, reliability, or accuracy challenges.

### Agent Coordination Patterns

#### 1. Parallel Tool Use
**Notebook**: [src/agents_parallelism/1_parallel_tool_use.ipynb](src/agents_parallelism/1_parallel_tool_use.ipynb)
- Execute multiple tool calls concurrently to reduce latency
- Demonstrates ~50% reduction in execution time vs sequential tool calling
- Real-world example: Fetching stock prices and news simultaneously
- **Use Case**: Any scenario requiring multiple independent API/tool calls

#### 2. Parallel Hypothesis Generation
**Notebook**: [src/agents_parallelism/2_parallel_hypothesis.ipynb](src/agents_parallelism/2_parallel_hypothesis.ipynb)
- Generate multiple solution approaches simultaneously
- Parallel exploration of solution space
- **Use Case**: Complex problem-solving, creative brainstorming

#### 3. Parallel Evaluation
**Notebook**: [src/agents_parallelism/3_parallel_evaluation.ipynb](src/agents_parallelism/3_parallel_evaluation.ipynb)
- Concurrent evaluation of multiple solutions against different criteria
- Accelerates quality assessment and decision-making
- **Use Case**: Multi-criteria decision making, A/B testing at scale

#### 4. Speculative Execution
**Notebook**: [src/agents_parallelism/4_speculative_execution.ipynb](src/agents_parallelism/4_speculative_execution.ipynb)
- Execute multiple likely paths in parallel before final decision
- Reduces perceived latency by pre-computing probable outcomes
- **Use Case**: Interactive applications, predictive workflows

#### 5. Hierarchical Agent Teams
**Notebook**: [src/agents_parallelism/5_hierarchical_agent_teams.ipynb](src/agents_parallelism/5_hierarchical_agent_teams.ipynb)
- Multi-level agent coordination with manager and worker agents
- Parallel execution at each hierarchy level
- **Use Case**: Complex projects requiring task decomposition and delegation

#### 6. Competitive Agent Ensemble
**Notebook**: [src/agents_parallelism/6_competitive_agent_ensemble.ipynb](src/agents_parallelism/6_competitive_agent_ensemble.ipynb)
- Multiple diverse agents compete in parallel, judge selects best output
- Uses different LLM models and personas for maximum diversity
- Demonstrates measurable quality improvements through competition
- **Use Case**: Content generation, code review, quality assurance

#### 7. Agent Assembly Line
**Notebook**: [src/agents_parallelism/7_agent_assembly_line.ipynb](src/agents_parallelism/7_agent_assembly_line.ipynb)
- Pipeline-based parallel processing with specialized agents
- Each stage processes multiple items concurrently
- **Use Case**: Batch processing, ETL pipelines, content workflows

#### 8. Decentralized Blackboard
**Notebook**: [src/agents_parallelism/8_decentralized_blackboard.ipynb](src/agents_parallelism/8_decentralized_blackboard.ipynb)
- Agents collaborate via shared state (blackboard pattern)
- Parallel agents read/write to common knowledge base
- **Use Case**: Collaborative problem-solving, distributed reasoning

#### 9. Redundant Execution
**Notebook**: [src/agents_parallelism/9_redundant_execution.ipynb](src/agents_parallelism/9_redundant_execution.ipynb)
- Execute same task with multiple agents, use voting for consensus
- Fault-tolerant parallel execution
- **Use Case**: Mission-critical decisions, error mitigation

### Advanced RAG Parallelism Patterns

#### 10. Parallel Query Expansion
**Notebook**: [src/agents_parallelism/10_parallel_query_expansion.ipynb](src/agents_parallelism/10_parallel_query_expansion.ipynb)
- Generate multiple query reformulations in parallel
- Enhances retrieval recall through diverse query perspectives
- **Use Case**: Semantic search, question answering systems

#### 11. Sharded Retrieval
**Notebook**: [src/agents_parallelism/11_sharded_retrieval.ipynb](src/agents_parallelism/11_sharded_retrieval.ipynb)
- Distribute document retrieval across multiple vector indices
- Parallel search over partitioned knowledge bases
- **Use Case**: Large-scale document collections, multi-domain RAG

#### 12. Hybrid Search Fusion
**Notebook**: [src/agents_parallelism/12_hybrid_search_fusion.ipynb](src/agents_parallelism/12_hybrid_search_fusion.ipynb)
- Parallel execution of semantic (vector) and lexical (keyword) search
- Fusion of results for maximum retrieval accuracy
- Combines best of both search paradigms
- **Use Case**: Enterprise search, technical documentation retrieval

#### 13. Parallel Context Preprocessing
**Notebook**: [src/agents_parallelism/13_parallel_context_preprocessing.ipynb](src/agents_parallelism/13_parallel_context_preprocessing.ipynb)
- Concurrent filtering and relevance checking of retrieved documents
- Reduces final context by 90%, improves latency by 25%
- Mitigates "lost in the middle" problem
- **Use Case**: High-recall RAG systems, cost optimization

#### 14. Multi-Hop Retrieval
**Notebook**: [src/agents_parallelism/14_multi_hop_retrieval.ipynb](src/agents_parallelism/14_multi_hop_retrieval.ipynb)
- Decompose complex queries into sub-questions
- Parallel retrieval for each sub-question
- Synthesis of multi-source evidence
- **Use Case**: Comparative analysis, research-oriented queries

## 🗄️ Vector Databases & RAG Systems

### Knowledge Graphs for RAG
**Notebooks**: [src/knowledge_graphs/](src/knowledge_graphs/)
- **Graph Construction**: [construct_kg.ipynb](src/knowledge_graphs/construct_kg.ipynb) - Build knowledge graphs from text
- **Graph Querying**: [query_kg.ipynb](src/knowledge_graphs/query_kg.ipynb) - Query patterns and traversal
- **Conversational Interfaces**: [chat_with_kg.ipynb](src/knowledge_graphs/chat_with_kg.ipynb) - Natural language graph interaction
- **Text-to-Graph RAG**: [text_for_rag.ipynb](src/knowledge_graphs/text_for_rag.ipynb) - Combining graphs with RAG

### Advanced Retriever Implementations

#### LlamaIndex Retrievers
**Notebook**: [src/vector_databases/retrievers/llamaindex_retrievers.ipynb](src/vector_databases/retrievers/llamaindex_retrievers.ipynb)
- **Vector Index Retriever**: Semantic similarity search with embeddings
- **BM25 Retriever**: Advanced keyword-based retrieval with TF-IDF
- **Document Summary Retriever**: Two-stage retrieval using summaries
- **Auto Merging Retriever**: Hierarchical chunking for context preservation
- **Recursive Retriever**: Following references and citations
- **Query Fusion Retriever**: Multiple fusion strategies (RRF, Relative Score)

#### LangChain Retrievers
**Notebook**: [src/vector_databases/retrievers/langchain_retrievers.ipynb](src/vector_databases/retrievers/langchain_retrievers.ipynb)
- **Multi-Query Retriever**: Multiple query perspectives for enhanced recall
- **Self-Query Retriever**: Natural language to structured query conversion
- **Parent Document Retriever**: Balancing chunk size for embeddings vs. context
- **Vector Store Retrievers**: MMR and similarity threshold implementations
- **Metadata Filtering**: Attribute-based search capabilities

### RAG Implementation Examples

#### RAG with HuggingFace
**Notebook**: [src/vector_databases/rag/rag_with_hf.ipynb](src/vector_databases/rag/rag_with_hf.ipynb)
- Complete RAG pipeline using HuggingFace models
- Embedding generation and vector storage
- Query processing and answer generation

#### RAG with PyTorch
**Notebook**: [src/vector_databases/rag/rag_with_torch.ipynb](src/vector_databases/rag/rag_with_torch.ipynb)
- Low-level RAG implementation with PyTorch
- Custom embedding models and similarity functions
- Fine-grained control over retrieval process

### Semantic Similarity & Vector Search
**Notebook**: [src/vector_databases/semantic_similarity/Semantic Similarity with FAISS.ipynb](src/vector_databases/semantic_similarity/Semantic%20Similarity%20with%20FAISS.ipynb)
- FAISS index construction and optimization
- Approximate nearest neighbor search
- Performance benchmarking and tuning

### Food Recommender RAG System
**Directory**: [src/vector_databases/food_recommender_rag/](src/vector_databases/food_recommender_rag/)
- Complete end-to-end RAG application example
- Interactive CLI chatbot with search capabilities
- Advanced filtering and metadata-based search
- Comprehensive documentation in [rag-cheatsheet.md](src/vector_databases/food_recommender_rag/rag-cheatsheet.md)

## 🧠 LangGraph & State Management

### Core LangGraph Components
**Notebook**: [src/langgraph/langgraph_components.ipynb](src/langgraph/langgraph_components.ipynb)
- StateGraph fundamentals and node definitions
- Conditional edges and routing logic
- Message passing and state updates

### Advanced LangGraph Patterns

#### Agentic Search
**Notebook**: [src/langgraph/agentic_search.ipynb](src/langgraph/agentic_search.ipynb)
- Building search agents with LangGraph
- Multi-step reasoning and information gathering
- State-based search orchestration

#### Persistence & Streaming
**Notebook**: [src/langgraph/persistence_and_streaming.ipynb](src/langgraph/persistence_and_streaming.ipynb)
- Checkpoint-based state persistence
- Streaming responses for better UX
- Resume interrupted workflows

#### Human-in-the-Loop
**Notebook**: [src/langgraph/human_in_loop.ipynb](src/langgraph/human_in_loop.ipynb)
- Interrupt workflows for human approval
- Conditional human intervention points
- Feedback integration patterns

#### Essay Writer Agent
**Notebook**: [src/langgraph/essay_writer_agent.ipynb](src/langgraph/essay_writer_agent.ipynb)
- Long-form content generation with reflection
- Multi-stage writing and editing workflow
- Quality improvement through iteration

#### ReAct Pattern
**Notebook**: [src/langgraph/ReAct.ipynb](src/langgraph/ReAct.ipynb)
- Reasoning and Acting pattern implementation
- Tool selection and execution cycles
- Iterative problem-solving workflow

### Memory Systems
**Files**: [src/memory/](src/memory/)
- **LangGraph Memory**: [memory_langgraph.py](src/memory/memory_langgraph.py) - Persistent memory with InMemoryStore
- **Memory Management**: [memory_management.py](src/memory/memory_management.py) - Core memory patterns
- Vector search and content filtering
- User context and conversation history

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

### Core Workflow Patterns
```bash
# Sequential processing
python src/workflows/1_sequential.py

# Multi-agent collaboration
python src/workflows/7_multi_agent.py

# Goal-driven code generation
python src/workflows/8_goal_setting.py
```

### Advanced Parallelism Notebooks
```bash
# Start Jupyter and explore parallelism patterns
jupyter notebook

# Recommended starting points:
# - src/agents_parallelism/1_parallel_tool_use.ipynb
# - src/agents_parallelism/6_competitive_agent_ensemble.ipynb
# - src/agents_parallelism/12_hybrid_search_fusion.ipynb
# - src/agents_parallelism/14_multi_hop_retrieval.ipynb
```

### RAG & Retrieval Systems
```bash
# Knowledge graph construction
jupyter notebook src/knowledge_graphs/construct_kg.ipynb

# Advanced retrievers
jupyter notebook src/vector_databases/retrievers/llamaindex_retrievers.ipynb
jupyter notebook src/vector_databases/retrievers/langchain_retrievers.ipynb

# RAG implementations
jupyter notebook src/vector_databases/rag/rag_with_hf.ipynb
```

### LangGraph Workflows
```bash
# Core components
jupyter notebook src/langgraph/langgraph_components.ipynb

# Advanced patterns
jupyter notebook src/langgraph/agentic_search.ipynb
jupyter notebook src/langgraph/human_in_loop.ipynb
jupyter notebook src/langgraph/essay_writer_agent.ipynb
```

### Simple Agent Examples
```bash
# Beginner-friendly tutorials
jupyter notebook "src/simple_agents/Build Interactive LLM Agents with Tools.ipynb"
jupyter notebook "src/simple_agents/Manual Tool-Calling Agent.ipynb"
```

## 📚 Key Dependencies

- **LangChain**: Core framework for LLM applications
- **LangGraph**: State management and workflow orchestration
- **LlamaIndex**: Advanced retrieval and indexing framework
- **CrewAI**: Multi-agent system framework
- **FastMCP**: Model Context Protocol implementation
- **ChromaDB**: Vector database for similarity search and RAG
- **HuggingFace Ecosystem**: Transformers, embeddings, and LLM integration
- **Sentence Transformers**: Advanced embedding models
- **BM25 & Retrieval Libraries**: Advanced keyword-based search
- **Google Generative AI**: Gemini model integration
- **Various LLM Providers**: OpenAI, Groq, Anthropic support
- **Scientific Computing**: NumPy, SciPy for mathematical operations

## 🎓 Learning Objectives

By exploring this repository, you will learn:

### Foundation
1. **Fundamental Agentic Patterns**: Core design principles for autonomous AI systems
2. **Workflow Orchestration**: Building complex, multi-step AI workflows with LangChain and LangGraph
3. **State Management**: Handling complex application state in AI systems
4. **Memory Management**: Implementing persistent and contextual memory

### Parallelism & Performance
5. **Parallel Execution Patterns**: 14 production-ready patterns for high-performance systems
6. **Concurrent Tool Calling**: Reducing latency through parallel API/function execution
7. **Competitive Ensembles**: Using agent diversity and voting for quality improvement
8. **Performance Optimization**: Benchmarking and optimizing AI system performance
9. **Fault Tolerance**: Building resilient systems with redundant execution

### Advanced RAG Techniques
10. **Multi-Hop Retrieval**: Decomposing complex queries for comprehensive answers
11. **Hybrid Search Fusion**: Combining semantic and lexical search strategies
12. **Parallel Context Processing**: Concurrent filtering for cost and latency reduction
13. **Query Expansion**: Parallel reformulation strategies for enhanced recall
14. **Sharded Retrieval**: Distributed search across multiple indices
15. **Knowledge Graphs**: Graph-based retrieval and reasoning

### Retrieval Systems
16. **Vector Databases**: FAISS, ChromaDB for similarity search
17. **Advanced Retrievers**: LlamaIndex and LangChain retrieval strategies
18. **RAG Architecture**: Building production-ready Retrieval-Augmented Generation pipelines
19. **Search & Filtering**: Multi-modal query processing and result refinement
20. **Fusion Techniques**: Combining retrieval methods for enhanced results

### Integration & Deployment
21. **Tool Integration**: Connecting AI agents to external systems and APIs
22. **Protocol Implementation**: Model Context Protocol (MCP) for standardized communication
23. **Multi-Agent Collaboration**: Designing systems where multiple AI agents coordinate
24. **Comparative Analysis**: Evaluating trade-offs between different architectures
25. **Production Best Practices**: Scalability, monitoring, and optimization strategies

## 📊 Performance Benefits Summary

The parallelism patterns in this repository demonstrate measurable improvements:

| Pattern | Performance Gain | Key Metric |
|---------|-----------------|------------|
| Parallel Tool Use | ~50% faster | Execution time reduction |
| Parallel Context Preprocessing | 90% token reduction | Cost optimization |
| Parallel Context Preprocessing | 25% faster | Generation latency |
| Competitive Ensemble | Quality improvement | Output diversity & selection |
| Multi-Hop Retrieval | Enhanced accuracy | Comprehensive multi-source answers |
| Hybrid Search Fusion | Better precision/recall | Combined semantic + lexical |

## 🗺️ Learning Path Recommendations

### Beginner Path
1. Start with [simple_agents/](src/simple_agents/) notebooks
2. Explore [workflows/](src/workflows/) patterns 1-3
3. Try basic [RAG examples](src/vector_databases/rag/)

### Intermediate Path
1. Study [LangGraph components](src/langgraph/langgraph_components.ipynb)
2. Explore [knowledge graphs](src/knowledge_graphs/)
3. Implement [advanced retrievers](src/vector_databases/retrievers/)
4. Try [parallel tool use](src/agents_parallelism/1_parallel_tool_use.ipynb)

### Advanced Path
1. Master all 14 [parallelism patterns](src/agents_parallelism/)
2. Implement [competitive ensembles](src/agents_parallelism/6_competitive_agent_ensemble.ipynb)
3. Build [multi-hop retrieval](src/agents_parallelism/14_multi_hop_retrieval.ipynb) systems
4. Optimize with [hybrid search fusion](src/agents_parallelism/12_hybrid_search_fusion.ipynb)

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different LLM providers
- Implement additional design patterns
- Extend existing examples
- Add new tool integrations
- Share performance benchmarks

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or discussions about agentic design patterns and parallelism techniques, please open an issue in the repository.

---

**Note**: This repository contains working implementations of theoretical concepts. Ensure you have appropriate API keys and rate limits configured before running examples. Performance metrics are based on example workloads and may vary based on your specific use case.