# LangChain Complete Study Notes

## 1. Models & I/O

### Chat Models & LLMs
- **ChatModels** - Primary interface for conversational AI
- **LLMs** - Traditional completion models
- **invoke()** - Single synchronous call
- **stream()** - Real-time streaming responses
- **batch()** - Process multiple inputs
- **ainvoke()** - Async single call
- **astream()** - Async streaming

### Message Types
- **SystemMessage** - System instructions
- **HumanMessage** - User input
- **AIMessage** - Model responses
- **ToolMessage** - Tool execution results

---

## 2. Prompts

### Templates
- **PromptTemplate** - Basic string templates with variables
- **ChatPromptTemplate** - Multi-message chat templates
- **MessagesPlaceholder** - Dynamic message insertion
- **FewShotPromptTemplate** - Examples-based prompting

### Template Methods
- `from_messages()` - Create from message list
- `from_template()` - Create from string
- `partial()` - Pre-fill variables
- Format with `{"variable": "value"}`

---

## 3. Output Parsers

### Core Parsers
- **StrOutputParser** - Plain text output
- **JsonOutputParser** - JSON responses
- **PydanticOutputParser** - Validated structured output
- **CommaSeparatedListOutputParser** - List parsing
- **RetryOutputParser** - Auto-retry on errors
- **OutputFixingParser** - Self-correcting parser

### Usage
```python
parser = PydanticOutputParser(pydantic_object=Model)
chain = prompt | llm | parser
```

---

## 4. Memory

### Memory Types
- **ConversationBufferMemory** - Store all messages
- **ConversationBufferWindowMemory** - Last K messages only
- **ConversationSummaryMemory** - Summarize old messages
- **ConversationTokenBufferMemory** - Token-limited buffer
- **VectorStoreRetrieverMemory** - Semantic similarity memory
- **ConversationEntityMemory** - Track entities

### Configuration
- `memory_key` - Variable name for history
- `return_messages` - Return as message objects
- `save_context()` - Save conversation turn
- `load_memory_variables()` - Retrieve history

---

## 5. Chains (Legacy & Modern)

### Legacy Chains
- **LLMChain** - Basic LLM + prompt chain
- **SequentialChain** - Multiple steps in sequence
- **SimpleSequentialChain** - Single input/output sequence
- **RetrievalQA** - QA with document retrieval
- **ConversationalRetrievalChain** - Conversational RAG

### LCEL (Modern Approach)
- **RunnableSequence** - Chain with `|` operator
- **RunnableParallel** - Parallel execution with `{}`
- **RunnableBranch** - Conditional routing
- **RunnablePassthrough** - Pass data through unchanged
- **RunnableLambda** - Custom functions

### LCEL Pattern
```python
chain = prompt | model | parser
result = chain.invoke({"input": "text"})
```

---

## 6. Runnables & LCEL

### Core Methods
- **invoke(input)** - Sync execution
- **stream(input)** - Streaming output
- **batch([inputs])** - Batch processing
- **ainvoke(input)** - Async execution
- **astream(input)** - Async streaming
- **abatch([inputs])** - Async batch

### Configuration Methods
- **with_config()** - Add configuration
- **with_fallbacks()** - Error fallback chains
- **with_retry()** - Retry logic
- **assign()** - Add new keys to passthrough
- **pick()** - Select specific keys

### Parallel Execution
```python
parallel = RunnableParallel(
    response1=chain1,
    response2=chain2
)
```

---

## 7. Document Loaders

### Common Loaders
- **TextLoader** - Plain text files
- **PyPDFLoader** - PDF documents
- **CSVLoader** - CSV files
- **DirectoryLoader** - Batch load directories
- **WebBaseLoader** - Web pages
- **UnstructuredFileLoader** - Multiple formats
- **JSONLoader** - JSON files
- **Docx2txtLoader** - Word documents

### Usage
```python
loader = PyPDFLoader("file.pdf")
documents = loader.load()
```

---

## 8. Text Splitters

### Splitter Types
- **RecursiveCharacterTextSplitter** - Intelligent splitting (recommended)
- **CharacterTextSplitter** - Simple character-based
- **TokenTextSplitter** - Token-based splitting
- **MarkdownTextSplitter** - Markdown structure-aware
- **PythonCodeTextSplitter** - Code-aware splitting
- **SemanticChunker** - Meaning-based chunks

### Configuration
- `chunk_size` - Maximum chunk length
- `chunk_overlap` - Overlap between chunks
- `separators` - Split delimiters

---

## 9. Embeddings

### Providers
- **OpenAIEmbeddings** - OpenAI models
- **HuggingFaceEmbeddings** - Open-source models
- **OllamaEmbeddings** - Local Ollama
- **CohereEmbeddings** - Cohere models
- **CacheBackedEmbeddings** - Cached embeddings

### Usage
```python
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(texts)
query_vector = embeddings.embed_query(query)
```

---

## 10. Vector Stores

### Popular Options
- **FAISS** - Fast similarity search (local)
- **Chroma** - Open-source database
- **Pinecone** - Managed cloud service
- **Qdrant** - Vector search engine
- **Weaviate** - Semantic search platform

### Operations
```python
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.add_documents(new_docs)
results = vectorstore.similarity_search(query, k=4)
vectorstore.save_local("path")
```

---

## 11. Retrievers

### Retriever Types
- **VectorStoreRetriever** - Similarity search
- **MultiQueryRetriever** - Generate multiple queries
- **ContextualCompressionRetriever** - Compress results
- **ParentDocumentRetriever** - Retrieve parent docs
- **EnsembleRetriever** - Combine multiple retrievers
- **SelfQueryRetriever** - Self-querying capability
- **TimeWeightedVectorStoreRetriever** - Time-aware

### Configuration
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
docs = retriever.invoke(query)
```

---

## 12. RAG Pipeline

### Basic RAG Flow
1. **Load** documents → DocumentLoader
2. **Split** into chunks → TextSplitter
3. **Embed** chunks → Embeddings
4. **Store** in vectorstore → VectorStore
5. **Retrieve** relevant docs → Retriever
6. **Generate** answer → LLM

### RAG Chain Pattern
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

### Advanced RAG
- **MultiQueryRetriever** - Multiple query variations
- **ContextualCompression** - Compress context
- **Re-ranking** - Re-rank retrieved docs
- **HyDE** - Hypothetical document embeddings

---

## 13. Tools & Toolkits

### Creating Tools
```python
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """Tool description"""
    return result
```

### Tool Types
- **Tool** - Basic tool wrapper
- **StructuredTool** - Structured input/output
- **create_retriever_tool** - Convert retriever to tool

### Common Tools
- **DuckDuckGoSearchRun** - Web search
- **WikipediaQueryRun** - Wikipedia queries
- **PythonREPLTool** - Execute Python code
- **Calculator** - Math calculations

### Toolkits
- **SQLDatabaseToolkit** - Database operations
- **VectorStoreToolkit** - Vector store operations
- **OpenAPIToolkit** - API interactions

---

## 14. Agents

### Agent Types
- **AgentExecutor** - Execute agent loops
- **Zero-shot ReAct** - Reasoning and acting
- **Conversational ReAct** - Chat with reasoning
- **OpenAI Functions Agent** - Function calling
- **Tool Calling Agent** - Dynamic tool selection

### Agent Pattern
```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": query})
```

---

## 15. Callbacks & Streaming

### Callback Handlers
- **StreamingStdOutCallbackHandler** - Stream to console
- **BaseCallbackHandler** - Custom callbacks
- **AsyncCallbackHandler** - Async callbacks

### Callback Events
- `on_llm_start` - LLM call starts
- `on_llm_new_token` - New token generated
- `on_llm_end` - LLM call completes
- `on_chain_start/end` - Chain execution

### Streaming Methods
```python
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)

# Advanced streaming
async for event in chain.astream_events(input):
    if event["event"] == "on_llm_new_token":
        print(event["data"]["chunk"], end="")
```

---

## 16. MCP (Model Context Protocol)

### Components
- **MCPToolkit** - MCP integration toolkit
- **MCPServer** - Connect to MCP servers
- **StdioMCPServer** - Stdio transport

### Operations
- `tools/list` - List available tools
- `tools/call` - Execute tool
- `resources/read` - Read resources
- `prompts/get` - Get prompts

---

## 17. Production Patterns

### Error Handling
```python
chain_with_fallback = chain.with_fallbacks([fallback_chain])
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1
)
```

### Configuration
```python
chain.with_config({
    "callbacks": [handler],
    "tags": ["production"],
    "metadata": {"version": "1.0"}
})
```

### Caching
- **CacheBackedEmbeddings** - Cache embeddings
- **InMemoryCache** - Memory caching
- **SQLiteCache** - Persistent caching

---

## 18. Key Architectural Patterns

### Pattern 1: Simple Chat
```
Prompt → LLM → StrOutputParser
```

### Pattern 2: RAG
```
Query → Retriever → Context + Prompt → LLM → Parser
```

### Pattern 3: Agent
```
Input → Agent (with Tools) → AgentExecutor → Output
```

### Pattern 4: Multi-Step Chain
```
Step1 → Step2 → Step3 (using RunnableSequence)
```

### Pattern 5: Parallel Processing
```
Input → {Chain1, Chain2, Chain3} → Combine → Output
```

---

## Quick Reference: Modern vs Legacy

### ✅ Modern (LCEL) - Use These
- `prompt | llm | parser` (LCEL chains)
- `RunnableSequence`, `RunnableParallel`
- `stream()`, `astream_events()`
- `@tool` decorator
- Pydantic models for output

### ⚠️ Legacy - Still Works
- `LLMChain`, `SequentialChain`
- Traditional callbacks
- Old chain classes

---

## Essential Production Checklist

- [ ] Use **ChatPromptTemplate** for structured prompts
- [ ] Implement **streaming** with `stream()` or `astream()`
- [ ] Add **error handling** with `with_fallbacks()`
- [ ] Use **RecursiveCharacterTextSplitter** for documents
- [ ] Implement **PydanticOutputParser** for structured output
- [ ] Add **ConversationMemory** for chat history
- [ ] Use **VectorStoreRetriever** for semantic search
- [ ] Implement **proper callbacks** for monitoring
- [ ] Cache embeddings with **CacheBackedEmbeddings**
- [ ] Use **@tool** decorator for custom tools

---

## Common Workflow Example

```python
# 1. Setup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2. Load & Split
loader = PyPDFLoader("doc.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(loader.load())

# 3. Create Vector Store
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 4. Build Chain
prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}"
)
llm = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute
response = chain.invoke("Your question here")
```

---

**Study Tips:**
- Master LCEL (`|` operator) for chain composition
- Practice RAG pipeline from end-to-end
- Understand retrievers and vector stores deeply
- Learn streaming for production apps
- Experiment with agents and tools
- Focus on error handling and fallbacks

**Remember:** LangChain prioritizes composability and modularity. Build small, reusable components and chain them together!