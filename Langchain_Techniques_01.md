# LangChain
llms_chat_model, prompts,output_parsers, memory, chains, messages.py", rag_retrievers,
embedding, retrievers, runnables, callbacks, text_splitter, tools_toolkits, document_loader,
streaming.py", mcp_client,
        
# LangGraph

state, nodes, edges, checkpointer", persistence", workflows


# Langchain all important technique names

# LangChain Essential Techniques Reference

## 1. Prompts (Core Techniques)

### Must-Know
- **ChatPromptTemplate** - Primary template for chat models
- **PromptTemplate** - Basic string-based prompts
- **MessagesPlaceholder** - Dynamic message insertion
- **FewShotPromptTemplate** - Few-shot learning with examples

### Important
- SystemMessagePromptTemplate
- HumanMessagePromptTemplate
- AIMessagePromptTemplate

---

## 2. Output Parsers (Core Techniques)

### Must-Know
- **PydanticOutputParser** - Structured output with validation
- **StrOutputParser** - Simple string output
- **JsonOutputParser** - JSON parsing

### Important
- CommaSeparatedListOutputParser
- RetryOutputParser
- OutputFixingParser

---

## 3. Memory (Core Techniques)

### Must-Know
- **ConversationBufferMemory** - Stores all messages
- **ConversationBufferWindowMemory** - Last K messages
- **ConversationSummaryMemory** - Summarized history

### Important
- VectorStoreRetrieverMemory
- ConversationTokenBufferMemory
- ConversationEntityMemory

---

## 4. Chains (Core Techniques)

### Must-Know (Legacy)
- **LLMChain** - Basic chain
- **SequentialChain** - Chain multiple steps
- **RetrievalQA** - Question answering with retrieval

### Must-Know (LCEL - Modern Approach)
- **RunnableSequence** - Chain with | operator
- **RunnableParallel** - Parallel execution
- **RunnableBranch** - Conditional routing
- **RunnablePassthrough** - Pass data through
- **RunnableLambda** - Custom functions

### Important
- ConversationalRetrievalChain
- StuffDocumentsChain
- MapReduceDocumentsChain

---

## 5. Messages (Core Techniques)

### Must-Know
- **HumanMessage** - User messages
- **AIMessage** - AI responses
- **SystemMessage** - System instructions
- **ToolMessage** - Tool execution results

### Important
- ChatMessageHistory
- FunctionMessage
- MessagesPlaceholder

---

## 6. Embeddings (Core Techniques)

### Must-Know
- **OpenAIEmbeddings** - OpenAI's embedding models
- **HuggingFaceEmbeddings** - Open-source models
- **CacheBackedEmbeddings** - Cached embeddings

### Important
- CohereEmbeddings
- VertexAIEmbeddings
- OllamaEmbeddings

---

## 7. Retrievers (Core Techniques)

### Must-Know
- **VectorStoreRetriever** - Basic similarity search
- **MultiQueryRetriever** - Multiple query generation
- **ContextualCompressionRetriever** - Compress retrieved docs

### Important
- EnsembleRetriever
- ParentDocumentRetriever
- SelfQueryRetriever
- TimeWeightedVectorStoreRetriever

---

## 8. Runnables / LCEL (Core Techniques)

### Must-Know
- **Runnable** - Base interface
- **pipe()** or **"|"** operator - Chain composition
- **invoke()** - Synchronous execution
- **stream()** - Streaming output
- **batch()** - Batch processing

### Must-Know (Async)
- **ainvoke()** - Async execution
- **astream()** - Async streaming
- **abatch()** - Async batch

### Important Methods
- with_config()
- with_fallbacks()
- with_retry()
- assign()
- pick()

---

## 9. Callbacks (Core Techniques)

### Must-Know
- **StreamingStdOutCallbackHandler** - Stream to stdout
- **BaseCallbackHandler** - Custom callbacks
- **on_llm_start** - LLM start event
- **on_llm_new_token** - Token streaming
- **on_llm_end** - LLM completion

### Important
- LangChainTracer
- OpenAICallbackHandler (token counting)
- AsyncCallbackHandler

---

## 10. Text Splitters (Core Techniques)

### Must-Know
- **RecursiveCharacterTextSplitter** - Intelligent splitting
- **CharacterTextSplitter** - Simple splitting
- **TokenTextSplitter** - Token-based splitting

### Important
- MarkdownTextSplitter
- PythonCodeTextSplitter
- HTMLTextSplitter
- SemanticChunker

---

## 11. Tools & Toolkits (Core Techniques)

### Must-Know
- **@tool** decorator - Create custom tools
- **Tool** - Basic tool wrapper
- **StructuredTool** - Structured input tools

### Common Tools
- DuckDuckGoSearchRun
- WikipediaQueryRun
- PythonREPLTool
- Calculator
- create_retriever_tool

### Important Toolkits
- SQLDatabaseToolkit
- VectorStoreToolkit
- OpenAPIToolkit

---

## 12. Document Loaders (Core Techniques)

### Must-Know
- **TextLoader** - Plain text files
- **PyPDFLoader** - PDF documents
- **CSVLoader** - CSV files
- **DirectoryLoader** - Batch loading
- **WebBaseLoader** - Web pages

### Important
- UnstructuredFileLoader
- JSONLoader
- Docx2txtLoader
- GitLoader
- NotionDBLoader

---

## 13. Streaming (Core Techniques)

### Must-Know
- **stream()** - Basic streaming
- **astream()** - Async streaming
- **astream_events()** - Detailed event streaming
- **StreamingStdOutCallbackHandler** - Console output

### Important Events
- on_llm_new_token
- on_llm_stream
- on_chain_stream

---

## 14. MCP (Model Context Protocol) (Core Techniques)

### Must-Know
- **MCPToolkit** - MCP integration
- **MCPServer** - Server connection
- **StdioMCPServer** - Stdio transport

### Important Operations
- tools/list
- tools/call
- resources/read
- prompts/get

---

## 15. Middleware (Core Techniques)

### Must-Know
- **with_config()** - Configuration middleware
- **with_fallbacks()** - Error fallback
- **with_retry()** - Retry logic
- **RunnableConfig** - Config propagation

### Important Patterns
- CallbackManager
- Tracing middleware
- Caching middleware
- Rate limiting

---

## Key Concepts Summary

### Modern LangChain (LCEL)
✅ **Use These:**
- Runnables + LCEL (with | operator)
- astream_events() for streaming
- Pydantic models for structured output
- @tool decorator for custom tools

### Legacy LangChain
⚠️ **Still Works But Consider Migrating:**
- Chain classes (LLMChain, SequentialChain)
- Traditional callback handlers
- Old-style tool definitions

---

## Most Critical for Production

1. **ChatPromptTemplate** - Structure your prompts
2. **RunnableSequence (LCEL)** - Build composable chains
3. **StreamingStdOutCallbackHandler** - Real-time output
4. **RecursiveCharacterTextSplitter** - Document chunking
5. **VectorStoreRetriever** - Semantic search
6. **PydanticOutputParser** - Structured responses
7. **ConversationBufferMemory** - Chat history
8. **with_fallbacks()** - Error handling
9. **astream_events()** - Production streaming
10. **@tool decorator** - Custom tool creation

---

## Quick Start Pattern

```python
# Modern LangChain pattern
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Build chain with LCEL
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

chain = prompt | ChatOpenAI() | StrOutputParser()

# Stream results
for chunk in chain.stream({"input": "Hello"}):
    print(chunk, end="")
```

---

*This reference focuses on the most important and frequently used techniques in LangChain. Master these first before exploring advanced features.*

