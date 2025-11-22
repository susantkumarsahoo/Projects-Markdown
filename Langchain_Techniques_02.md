# LangChain Techniques - Complete Reference Guide

## 1. LLMs & Chat Models

### Core Components
- **ChatModels**: Primary interface for chat-based language models
- **LLMs**: Traditional language model interface
- **llms_chat_model**: Unified model interface
- **Model I/O**: Input/output handling for models

### Prompts & Messages
- **Prompt Templates**: Reusable prompt structures
- **ChatPromptTemplate**: Chat-specific prompt templates
- **PromptValue**: Serialized prompt representation
- **Messages**: 
  - `BaseMessage`: Base class for all messages
  - `HumanMessage`: User input messages
  - `AIMessage`: AI response messages
  - `SystemMessage`: System instructions

### Model Operations
- **Message Conversion utilities**: Convert between message formats
- **Model Invocation**: 
  - `invoke`: Single synchronous call
  - `stream`: Streaming responses
  - `batch`: Batch processing
- **Structured Outputs**: Type-safe output generation

---

## 2. Prompts & Templates

### Template Types
- **PromptTemplate**: Basic string-based templates
- **ChatPromptTemplate**: Multi-message templates
- **Few-shot prompts**: Examples-based prompting
- **Dynamic Prompts**: Runtime-generated prompts
- **Prompt Serialization**: Save/load prompt configurations

### Advanced Features
- **Prompt Variables**: Dynamic variable substitution
- **Partial Prompting**: Pre-fill some variables
- **Prompt Composition**: Combine multiple prompts
- **System/Human/AI messages**: Role-specific message templates

---

## 3. Output Parsers

### Parser Types
- **StrOutputParser**: Plain string output
- **JsonOutputParser**: JSON-formatted output
- **PydanticOutputParser**: Type-validated output with Pydantic models
- **Structured Output Parser**: Custom structured formats
- **RegexParser**: Pattern-based parsing
- **Enum Output Parser**: Enumerated value parsing
- **RetryWithErrorOutputParser**: Automatic retry on parsing errors
- **Output Fixing Parser**: Self-correcting parser

---

## 4. Memory (Conversation Memory)

### Memory Types
- **BufferMemory**: Store all conversation history
- **BufferWindowMemory**: Keep last N messages
- **ConversationSummaryMemory**: Summarize older conversations
- **ConversationKGMemory**: Knowledge graph-based memory
- **EntityMemory**: Track entities across conversations
- **VectorStoreRetrieverMemory**: Semantic similarity-based memory
- **ConversationTokenBufferMemory**: Token-limited buffer

### Storage Backends
- **ChatMessageHistory**: In-memory storage
- **FileChatMessageHistory**: File-based persistence
- **Postgres/SQL Memory**: Database-backed memory

---

## 5. Chains (Core Logic Units)

### Basic Chains
- **LLMChain**: Simple LLM call with prompt
- **SequentialChain**: Execute chains in sequence
- **SimpleSequentialChain**: Sequential with single input/output
- **ConversationChain**: Conversation management with memory

### Advanced Chains
- **RetrievalQA**: Question-answering with retrieval
- **Map-Reduce Chain**: Parallel processing then combine
- **Refine Chain**: Iterative refinement of outputs
- **Hypothetical Document Embeddings (HyDE)**: Generate hypothetical answers
- **SummarizationChain**: Document summarization
- **RouterChain**: Dynamic chain routing
- **TransformChain**: Data transformation
- **LLMMathChain**: Mathematical reasoning
- **Tool-Using Chains**: Integrate external tools

---

## 6. Runnables (LangChain Expression Language)

### Core Runnables
- **Runnable**: Base interface for all runnables
- **RunnableLambda**: Custom function wrapper
- **RunnableSequence**: Chain runnables sequentially
- **RunnableParallel**: Execute runnables in parallel
- **RunnablePassthrough**: Pass data through unchanged
- **RunnableBranch**: Conditional execution
- **RunnableMap**: Apply function to multiple inputs

### Advanced Features
- **RunnableConfigurable**: Runtime configuration
- **RunnableWithFallbacks**: Fallback on errors
- **RunnableWithRetry**: Automatic retry logic

### Invocation Methods
- `invoke`: Synchronous execution
- `stream`: Streaming execution
- `batch`: Batch processing
- `ainvoke`: Async execution

---

## 7. Document Loading

### Document Loaders
- **DocumentLoader**: Base document loader
- **TextLoader**: Plain text files
- **DirectoryLoader**: Load entire directories
- **WebBaseLoader**: Web page content
- **PyPDFLoader**: PDF documents
- **CSVLoader**: CSV files
- **Unstructured loaders**: Various unstructured formats
- **HTML/XML/JSON loaders**: Structured document formats
- **NotebookLoader**: Jupyter notebooks (.ipynb)
- **YouTubeLoader**: YouTube transcripts

---

## 8. Text Splitters

### Splitter Types
- **RecursiveCharacterTextSplitter**: Smart splitting with recursion
- **CharacterTextSplitter**: Split by character count
- **TokenTextSplitter**: Split by token count
- **Semantic Text Splitter**: Meaning-based splitting
- **MarkdownHeaderTextSplitter**: Split by Markdown structure
- **NLTK/SpaCy Splitters**: NLP-based splitting
- **Language-aware splitters**: Language-specific splitting

---

## 9. Embeddings

### Embedding Providers
- **OpenAIEmbeddings**: OpenAI embedding models
- **HuggingFaceEmbeddings**: HuggingFace models
- **GoogleEmbeddings**: Google embedding models
- **SentenceTransformerEmbeddings**: Sentence transformers
- **OllamaEmbeddings**: Local Ollama embeddings
- **FastEmbedEmbeddings**: Fast embedding generation

### Embedding Types
- **Self-query Embeddings**: Query-specific embeddings
- **Document Embeddings**: Document representation
- **Query Embeddings**: Query representation

---

## 10. Vectorstores (Retrievers + Storage)

### Vector Database Options
- **FAISS**: Facebook AI Similarity Search
- **Chroma**: Open-source embedding database
- **Pinecone**: Managed vector database
- **Qdrant**: Vector search engine
- **Weaviate**: Vector search and storage
- **Milvus**: Distributed vector database
- **Elasticsearch**: Search and analytics engine
- **SQL Vectorstores**: SQL-based vector storage
- **MongoDB Atlas**: MongoDB vector search
- **Redis Vectorstore**: Redis-based vector storage

---

## 11. Retrievers

### Standard Retrievers
- **VectorStoreRetriever**: Vector similarity search
- **MultiQueryRetriever**: Multiple query variations
- **ParentDocumentRetriever**: Retrieve parent documents
- **ContextualCompressionRetriever**: Compress context
- **SelfQueryRetriever**: Self-querying retrieval
- **MultiVectorRetriever**: Multiple vector strategies
- **BM25Retriever**: Keyword-based retrieval
- **EnsembleRetriever**: Combine multiple retrievers
- **Time-weighted Vector Retriever**: Time-aware retrieval

### RAG Techniques
- **RAGRetrievers**: Retrieval-Augmented Generation
- **HyDE Retriever**: Hypothetical Document Embeddings
- **FusionRetrieval**: Fusion of retrieval methods
- **CompressionRetrievers**: Context compression
- **Long-context RAG**: Handle long contexts
- **Memory-augmented RAG**: RAG with memory

---

## 12. Tools & Toolkits

### Individual Tools
- **Tool**: Base tool interface
- **StructuredTool**: Structured input/output tool
- **PythonREPLTool**: Python code execution
- **RequestsGetTool**: HTTP GET requests
- **RequestsPostTool**: HTTP POST requests
- **Search Tools**: Web search integration
- **Calculator Tool**: Mathematical calculations
- **Database Tools**: Database operations
- **Vectorstore Tools**: Vector database operations
- **Filesystem tools**: File system operations

### Toolkits (Tool Collections)
- **SQLToolkit**: SQL database operations
- **OpenAIToolkit**: OpenAI-specific tools
- **VectorStoreToolkit**: Vector store operations
- **DatabaseToolkit**: Database management
- **FileToolkit**: File operations
- **BrowserToolkit**: Web browser automation

---

## 13. Agents

### Agent Types
- **AgentExecutor**: Core agent execution engine
- **Zero-shot ReAct Agent**: Reasoning and acting without examples
- **Conversational ReAct Agent**: Conversational reasoning agent
- **Plan-and-Execute Agent**: Planning before execution
- **OpenAI Function Agent**: OpenAI function calling
- **Tool Calling Agent**: Dynamic tool selection
- **Self-query Agent**: Self-querying capabilities
- **Agent with memory**: Memory-enabled agents

### Multi-Agent Systems
- **Multi-agent systems**: Multiple coordinated agents
- **Agent Supervisor**: Agent coordination and supervision

---

## 14. Callbacks

### Callback Types
- **StreamingStdOutCallbackHandler**: Stream output to stdout
- **AsyncCallbackHandlers**: Asynchronous callbacks
- **Tracer Callback**: Execution tracing
- **Run Logs**: Logging execution details
- **Langsmith Integration**: LangSmith monitoring
- **Token Counting Callback**: Track token usage
- **Custom callback handlers**: User-defined callbacks

---

## 15. Streaming

### Streaming Features
- **LLM streaming**: Stream LLM responses
- **ChatModel streaming**: Stream chat responses
- **Partial responses**: Handle incomplete responses
- **Token-by-token generation**: Real-time token streaming
- **Async streaming**: Asynchronous streaming

---

## 16. Middleware & Observability

### Monitoring & Debugging
- **mcp_client**: Model Context Protocol
- **Middleware**: Request/response processing
- **Request/Response Interceptors**: Intercept and modify calls
- **Telemetry**: Performance metrics
- **Tracing**: Execution tracing
- **LangSmith**: Official monitoring platform
- **OpenTelemetry**: Standard telemetry framework
- **Run Monitoring**: Monitor execution runs
- **Debugging Tools**: Development debugging utilities

---

## 17. RAG Pipelines

### Pipeline Components
- **Indexing**: Document indexing
- **Document Transformation**: Transform documents
- **Document Compression**: Compress context
- **Retrieval**: Retrieve relevant documents
- **Re-ranking**: Re-rank retrieved documents
- **Answer generation**: Generate final answers
- **Guardrail pipelines**: Safety and quality checks

### Advanced RAG
- **RAG Fusion**: Combine multiple RAG strategies
- **Advanced RAG**: 
  - GraphRAG: Knowledge graph-based RAG
  - Multi-Hop RAG: Multi-step reasoning

---

## 18. Deployment & Integrations

### Deployment Options
- **FastAPI integration**: REST API deployment
- **Streamlit integration**: Interactive web apps
- **Gradio integration**: ML model interfaces
- **LangServe**: LangChain deployment server
- **LangGraph**: Complex workflow orchestration

### Platform Integrations
- **OpenAI**: OpenAI model integration

---

## Use Cases

This comprehensive list covers all essential LangChain concepts for:

✅ **Chatbot development**: Build conversational AI
✅ **RAG pipelines**: Retrieval-Augmented Generation
✅ **Modular architecture**: Clean, maintainable code
✅ **Industry-ready LLM projects**: Production deployments
✅ **Agentic workflows**: Autonomous agent systems
✅ **Production deployments**: Scalable applications

---

## Next Steps

Want more? I can provide:
- 📁 Modular folder structure using these techniques
- 🏗️ Complete production-ready LangChain + RAG project architecture
- 🔄 LangGraph techniques list (for complex workflows)

Just ask for "folder structure" or "LangGraph techniques"!