✅ ALL IMPORTANT LANGCHAIN TECHNIQUE NAMES (Organized + Complete)

# 1. LLMs & Chat Models

### ChatModels, LLMs, llms_chat_model, Model I/O, Prompt Templates, ChatPromptTemplate
### PromptValue, Messages (BaseMessage, HumanMessage, AIMessage, SystemMessage)
### Message Conversion utilities, Model Invocation (invoke, stream, batch), Structured Outputs

# 2. Prompts & Templates, 
### PromptTemplate, ChatPromptTemplate, Few-shot prompts, Dynamic Prompts, Prompt Serialization
### Prompt Variables, Partial Prompting, Prompt Composition, System/Human/AI messages

# 3. Output Parsers
### StrOutputParser, JsonOutputParser, PydanticOutputParser, Structured Output Parser, RegexParser
### Enum Output Parser, RetryWithErrorOutputParser, Output Fixing Parser

# 4. Memory (Conversation Memory)
### BufferMemory, BufferWindowMemory, ConversationSummaryMemory, ConversationKGMemory
### EntityMemory, VectorStoreRetrieverMemory, ChatMessageHistory, FileChatMessageHistory
### Postgres/SQL Memory, ConversationTokenBufferMemory

# 5. Chains (Core Logic Units)


LLMChain


SequentialChain


SimpleSequentialChain


ConversationChain


RetrievalQA


Map-Reduce Chain


Refine Chain


Hypothetical Document Embeddings (HyDE)


SummarizationChain


RouterChain


TransformChain


LLMMathChain


Tool-Using Chains



6. Runnables (LangChain Expression Language)


Runnable


RunnableLambda


RunnableSequence


RunnableParallel


RunnablePassthrough


RunnableBranch


RunnableMap


RunnableConfigurable


RunnableWithFallbacks


RunnableWithRetry


runnable.invoke/stream/batch/ainvoke



7. Document Loading


DocumentLoader


TextLoader


DirectoryLoader


WebBaseLoader


PyPDFLoader


CSVLoader


Unstructured loaders


HTML/XML/JSON loaders


NotebookLoader (.ipynb)


YouTubeLoader



8. Text Splitters


RecursiveCharacterTextSplitter


CharacterTextSplitter


TokenTextSplitter


Semantic Text Splitter


MarkdownHeaderTextSplitter


NLTK/SpaCy Splitters


Language-aware splitters



9. Embeddings


OpenAIEmbeddings


HuggingFaceEmbeddings


GoogleEmbeddings


SentenceTransformerEmbeddings


OllamaEmbeddings


FastEmbedEmbeddings


Self-query Embeddings


Document Embeddings


Query Embeddings



10. Vectorstores (Retrievers + Storage)


FAISS


Chroma


Pinecone


Qdrant


Weaviate


Milvus


Elasticsearch


SQL Vectorstores


MongoDB Atlas


Redis Vectorstore



11. Retrievers
🔹 Standard Retrievers


VectorStoreRetriever


MultiQueryRetriever


ParentDocumentRetriever


ContextualCompressionRetriever


SelfQueryRetriever


MultiVectorRetriever


BM25Retriever


EnsembleRetriever


Time-weighted Vector Retriever


🔹 RAG Techniques


RAGRetrievers


HyDE Retriever


FusionRetrieval


CompressionRetrievers


Long-context RAG


Memory-augmented RAG



12. Tools & Toolkits
🔹 Tools


Tool


StructuredTool


PythonREPLTool


RequestsGetTool


RequestsPostTool


Search Tools


Calculator Tool


Database Tools


Vectorstore Tools


Filesystem tools


🔹 Toolkits


SQLToolkit


OpenAIToolkit


VectorStoreToolkit


DatabaseToolkit


FileToolkit


BrowserToolkit



13. Agents


AgentExecutor


Zero-shot ReAct Agent


Conversational ReAct Agent


Plan-and-Execute Agent


OpenAI Function Agent


Tool Calling Agent


Self-query Agent


Agent with memory


Multi-agent systems


Agent Supervisor



14. Callbacks


StreamingStdOutCallbackHandler


AsyncCallbackHandlers


Tracer Callback


Run Logs


Langsmith Integration


Token Counting Callback


Custom callback handlers



15. Streaming


LLM streaming


ChatModel streaming


Partial responses


Token-by-token generation


Async streaming



16. Middleware / Observability


mcp_client (Model Context Protocol)


Middleware


Request/Response Interceptors


Telemetry


Tracing


LangSmith


OpenTelemetry


Run Monitoring


Debugging Tools



17. RAG Pipelines


Indexing


Document Transformation


Document Compression


Retrieval


Re-ranking


Answer generation


Guardrail pipelines


RAG Fusion


Advanced RAG (GraphRAG, Multi-Hop RAG)



18. Deployment & Integrations


FastAPI integration


Streamlit integration


Gradio integration


LangServe


LangGraph


OpenAI



✔️ This list includes every important LangChain concept needed for:


Chatbot development


RAG pipelines


Modular architecture


Industry-ready LLM projects


Agentic workflows


Production deployments



If you want, I can also provide:
✅ A modular folder structure using all these techniques
✅ A complete production-ready LangChain + RAG project architecture
✅ A list of LangGraph techniques (if you need them too)
Just tell me “Make folder structure” or "Give LangGraph techniques".