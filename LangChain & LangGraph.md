# 🦜 LangChain & LangGraph Reference Guide

> A comprehensive collection of techniques, components, and patterns for building AI applications

---

## 📚 Table of Contents

- [LangChain Techniques](#langchain-techniques)
- [LangGraph Techniques](#langgraph-techniques)
- [Quick Reference](#quick-reference)
- [Best Practices](#best-practices)

---

## 🦜 LangChain Techniques

### Core Components

| Component | Description |
|-----------|-------------|
| **LLMs** | Large Language Model interfaces |
| **Chat Models** | Conversational AI model wrappers |
| **Prompt Templates** | Reusable prompt structures |
| **LCEL** | LangChain Expression Language for chaining |
| **Chains** | Sequential, Router, Conditional workflows |
| **Agents & Tools** | Autonomous decision-making components |

### Prompting Techniques

**Basic**
- Prompt Templates
- Chat Prompt Templates
- System/Role Prompting
- Dynamic Prompting

**Advanced**
- Few-Shot Prompting
- Zero-Shot Prompting
- Retrieval-Augmented Prompting
- Context Injection

**Structured Output**
- Function Calling
- Tool Calling
- Structured Output Prompting

### Memory Types

| Memory Type | Use Case | Retention Strategy |
|-------------|----------|-------------------|
| **ConversationBufferMemory** | Store all messages | Full history |
| **ConversationBufferWindowMemory** | Recent messages only | Sliding window |
| **ConversationTokenBufferMemory** | Token-limited storage | Token count |
| **ConversationSummaryMemory** | Summarized history | Summarization |
| **ConversationSummaryBufferMemory** | Hybrid approach | Summary + recent |
| **VectorStoreRetrieverMemory** | Semantic search | Vector similarity |
| **EntityMemory** | Entity-focused tracking | Entity extraction |

**Storage Backends:** PostgreSQL, Redis, ChatMessageHistory

### Retrieval & RAG

**Vector Stores:** Chroma, FAISS, Pinecone, Milvus, Weaviate

**Embeddings:** OpenAI, HuggingFace, Cohere, Custom Models

**Retrievers:** MultiQuery, Contextual Compression, Parent Document, Ensemble, BM25

**Advanced RAG Patterns**

| Pattern | Description |
|---------|-------------|
| **Self-RAG** | Self-correcting retrieval for improved accuracy |
| **RAG Fusion** | Multiple query fusion for better coverage |
| **Document Compression** | Context compression to reduce tokens |
| **Re-ranking** | Result optimization for better relevance |

**Document Processing:** PDF/CSV/HTML Loaders, Text Splitters (Recursive, Character, Semantic), Metadata Extraction

### Agent Types

**Core Agents**
- ReAct Agent
- Tool-Calling Agents
- OpenAI Function Agent
- Conversational Agent
- Self-Ask Agent
- Planner-Executor Agents

**Specialized Agents**
- Zero-Shot Agent
- Structured Output Agent
- JSON Agent
- SQL Agent
- VectorStore Agent

**Execution Pattern:** Thought → Action → Observation → Thought → ... → Final Answer

### Tools & Integration

| Category | Tools | Purpose |
|----------|-------|---------|
| **Computation** | Python REPL, Calculator | Execute code & math |
| **Search** | Search Tools, API Tools | Information retrieval |
| **Data** | SQL Tools, DataFrame Tools | Data manipulation |
| **Web** | Browser Tools, HTTP Tools | Web interaction |
| **File** | File Tools, Document Tools | File operations |

**Tool Creation:** `@tool` decorator, `StructuredTool` class, Custom Toolkit Building

### Output Processing

**Parsers:** Output Parsers, Structured Output Parser, JSON Output Parser, PydanticOutputParser, RetryOutputParser

**Validation:** Schema Validation, Guardrails, Type Checking, Error Recovery

### Workflow & LCEL

**Runnable Components**

| Component | Purpose | Pattern |
|-----------|---------|---------|
| **RunnableSequence** | Linear execution | A → B → C |
| **RunnableParallel** | Parallel execution | A ∥ B ∥ C |
| **RunnableBranch** | Conditional routing | If-then-else |
| **Retry & Fallback** | Error handling | Resilience |

**Execution Modes:** Streaming, Batch Processing, Async Processing

**Observability:** Tracing & Instrumentation, Performance Monitoring, Debug Logging

### Deployment & Operations

**Deployment:** LangServe (API serving), Endpoint Hosting, Container Deployment

**Operations:** Rate Limiting, Caching (Redis, Memory), Monitoring (LangSmith), Cost Tracking

---

## 🕸️ LangGraph Techniques

### Core Graph Concepts

| Concept | Description |
|---------|-------------|
| **State Graph** | Directed graph with state for workflow orchestration |
| **Node** | Execution unit (processing step) |
| **Edge** | Connection between nodes for flow control |
| **State Management** | Shared data across nodes |
| **Checkpointing** | Save graph state for recovery & replay |
| **Branching** | Multiple paths for conditional logic |
| **Loops** | Iterative execution for repeated operations |
| **Subgraphs** | Nested graphs for modular design |

### Node Types

**Execution Nodes:** Tool Node, LLM Node, Transform Node, Message Node

**Control Nodes:** Router Node, Condition Node, Human-in-the-Loop Node, Retry Node, Error Node

### State Management

| Technique | Behavior | Use Case |
|-----------|----------|----------|
| **Typed State** | Schema-enforced state | Type safety |
| **Shared State** | Global state access | Cross-node data |
| **Merged State** | State combination | Data aggregation |
| **Additive State** | Append-only updates | History tracking |
| **Snapshot State** | Point-in-time capture | Versioning |
| **Checkpoint Restore** | State recovery | Error recovery |

### Control Flow

**Routing Patterns:** If-Else Routing, Dynamic Routing, Multi-path Execution, Conditional Branching

**Resilience Patterns:** Loops (While/For), Retry Logic, Fallback Logic, Timeout Control, Guardrail Validation

**Flow Pattern:** Start → Condition → [Path A | Path B] → Merge → End

### Agentic Workflows

| Pattern | Best For |
|---------|----------|
| **Multi-Agent Orchestration** | Complex tasks requiring coordination |
| **Planner-Executor Graph** | Strategic workflows with planning phase |
| **Tool-Calling Agents** | Action-oriented tasks |
| **Routing Agents** | Specialized routing between agents |
| **Supervisor-Worker Graphs** | Hierarchical team coordination |
| **Task Decomposition** | Breaking down large projects |

### RAG + Graph

**Retrieval Patterns:** Modular RAG Graph, Iterative Retrieval, Multi-Retriever Graph, Self-RAG Graph

**Validation Nodes:** Reality Check, Verification, Context Revalidation, Evidence Collection

**Advanced Flow:** Query → Retrieve → Validate → Re-retrieve (if needed) → Generate → Verify

### Error Handling

| Technique | Recovery Strategy |
|-----------|-------------------|
| **Error Nodes** | Catch & process errors |
| **Exception Routing** | Route errors to handlers |
| **Fallback Node** | Backup strategy |
| **Tool Failure Recovery** | Retry or use alternate tool |
| **Response Revalidation** | Quality assurance checks |
| **Automatic Node Retry** | Handle transient failures |

### Evaluation & Monitoring

**Debugging:** Checkpoint Inspection, State Diffing, Execution Replay, Step-by-step Analysis

**Observability:** Graph Tracing, Performance Metrics, LangSmith Integration, Execution Logs

**Key Metrics:** Node execution time, State transitions, Error rates, Token usage, Success rates

---

## 🎯 Quick Reference

| Use Case | LangChain | LangGraph |
|----------|-----------|-----------|
| Simple Q&A | Chain + LLM | Not needed |
| Multi-step reasoning | Sequential Chain | State Graph |
| Tool usage | Agent + Tools | Tool Node + Router |
| Conversational | Memory + Chain | State Graph + Memory |
| Complex workflows | LCEL + Chains | Multi-node Graph |
| Human-in-the-loop | Not native | HITL Node |
| Error recovery | Retry/Fallback | Error Nodes |
| RAG pipeline | RAG Chain | RAG Graph |

---

## 📖 Best Practices

### LangChain
✅ Use LCEL for composable chains  
✅ Choose appropriate memory based on context size  
✅ Cache embeddings for efficiency  
✅ Monitor token usage in production  
✅ Implement proper error handling  

### LangGraph
✅ Define clear state schemas  
✅ Use checkpointing for long-running workflows  
✅ Implement human-in-the-loop for critical decisions  
✅ Trace execution paths for debugging  
✅ Design for failure with error nodes  

---

## 🔗 Resources

- **LangChain Docs:** https://python.langchain.com/
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **LangSmith:** https://smith.langchain.com/

---

*Last Updated: November 2025*




# 🎵 Advanced Playlist: LangChain & LangGraph Techniques

> A structured reference playlist of advanced techniques and features for building production-grade LLM applications with **LangChain** and **LangGraph**.

---

## Table of Contents

- [LangChain — Advanced Techniques](#-langchain--advanced-techniques)
  - [Chains & Composition](#1-chains--composition)
  - [Retrieval-Augmented Generation (RAG)](#2-retrieval-augmented-generation-rag)
  - [Memory & State Management](#3-memory--state-management)
  - [Tools & Agents](#4-tools--agents)
  - [Callbacks & Observability](#5-callbacks--observability)
  - [Prompt Engineering](#6-prompt-engineering)
  - [Document Loaders & Transformers](#7-document-loaders--transformers)
  - [Output Parsers](#8-output-parsers)
  - [Embeddings & Vector Stores](#9-embeddings--vector-stores)
  - [Streaming](#10-streaming)
- [LangGraph — Advanced Techniques](#-langgraph--advanced-techniques)
  - [Graph Architecture](#1-graph-architecture)
  - [State Management](#2-state-management)
  - [Control Flow](#3-control-flow)
  - [Persistence & Checkpointing](#4-persistence--checkpointing)
  - [Multi-Agent Systems](#5-multi-agent-systems)
  - [Human-in-the-Loop](#6-human-in-the-loop)
  - [Streaming & Observability](#7-streaming--observability)
  - [Deployment & Production](#8-deployment--production)
- [Integration Patterns](#-integration-patterns)

---

## 🔗 LangChain — Advanced Techniques

### 1. Chains & Composition

| # | Technique | Description |
|---|-----------|-------------|
| 01 | **LCEL (LangChain Expression Language)** | Declarative pipe-based (`\|`) syntax to compose runnables into chains with lazy evaluation |
| 02 | **RunnableSequence** | Chain multiple runnables sequentially, passing output of one as input to the next |
| 03 | **RunnableParallel** | Execute multiple runnables in parallel and merge their outputs into a single dict |
| 04 | **RunnableBranch** | Conditional routing — select which branch to run based on a predicate function |
| 05 | **RunnableLambda** | Wrap any Python callable as a first-class runnable in an LCEL chain |
| 06 | **RunnablePassthrough** | Pass input through unchanged; useful for injecting context or merging with other outputs |
| 07 | **RunnableWithFallbacks** | Attach fallback runnables that trigger on exceptions, enabling resilient chains |
| 08 | **RunnableRetry** | Auto-retry a runnable on transient errors with configurable backoff |
| 09 | **RunnableMap** | Apply the same runnable over each element of an iterable input |
| 10 | **Custom Chain Composition** | Build reusable domain-specific chains by subclassing `Chain` with custom `_call` logic |

---

### 2. Retrieval-Augmented Generation (RAG)

| # | Technique | Description |
|---|-----------|-------------|
| 11 | **MultiQueryRetriever** | Generate multiple query reformulations and merge results to overcome single-query limitations |
| 12 | **ContextualCompressionRetriever** | Post-process retrieved documents with a compressor to extract only relevant passages |
| 13 | **EnsembleRetriever** | Combine multiple retrievers (e.g., BM25 + dense vector) using reciprocal rank fusion |
| 14 | **ParentDocumentRetriever** | Index small child chunks for retrieval but return full parent documents for context |
| 15 | **SelfQueryRetriever** | Auto-generate structured metadata filters from natural language queries |
| 16 | **TimeWeightedVectorStoreRetriever** | Decay retrieval scores based on recency to bias towards fresh documents |
| 17 | **Hypothetical Document Embeddings (HyDE)** | Embed a generated hypothetical answer and use it as the retrieval query |
| 18 | **Reranking with Cross-Encoders** | Re-score retrieved candidates with a more powerful cross-encoder model before returning top-k |
| 19 | **Recursive RAG / Iterative Retrieval** | Feed retrieved content back into retrieval loops to progressively refine answers |
| 20 | **RAPTOR (Recursive Abstractive Processing)** | Build a hierarchical tree of summaries and retrieve at multiple abstraction levels |
| 21 | **Fusion RAG** | Reciprocal rank fusion across multiple diverse retrieval strategies |
| 22 | **Graph RAG** | Combine knowledge graph traversal with vector retrieval for relationship-aware RAG |
| 23 | **Adaptive RAG** | Dynamically choose retrieval strategy (local/global/no retrieval) based on query type |

---

### 3. Memory & State Management

| # | Technique | Description |
|---|-----------|-------------|
| 24 | **ConversationBufferMemory** | Maintain the full raw conversation history in the context window |
| 25 | **ConversationBufferWindowMemory** | Sliding window over the last N conversation turns to control token cost |
| 26 | **ConversationSummaryMemory** | Progressively summarize older turns to compress history without losing key facts |
| 27 | **ConversationSummaryBufferMemory** | Hybrid: buffer recent turns verbatim, summarize older turns |
| 28 | **EntityMemory** | Extract and persist named entities (people, places, concepts) with structured facts |
| 29 | **VectorStoreRetrieverMemory** | Store every exchange as an embedding; retrieve relevant past turns by semantic search |
| 30 | **ReadOnlySharedMemory** | Expose a read-only view of memory across multiple chains without write conflicts |
| 31 | **Custom Memory Classes** | Implement `BaseMemory` interface to persist state in any external store (Redis, SQL, etc.) |

---

### 4. Tools & Agents

| # | Technique | Description |
|---|-----------|-------------|
| 32 | **Structured Tool (StructuredTool)** | Define tools with Pydantic schemas for multi-argument, type-safe tool invocation |
| 33 | **Tool Decorators (`@tool`)** | Rapidly define tools from Python functions with auto-extracted docstring descriptions |
| 34 | **Toolkit Pattern** | Bundle related tools into a cohesive `BaseToolkit` for domain-specific agent capabilities |
| 35 | **OpenAI Tools / Function Calling Agent** | Leverage native model function-calling APIs for reliable, schema-constrained tool use |
| 36 | **ReAct Agent** | Thought → Action → Observation loop for step-by-step reasoning before tool calls |
| 37 | **Plan-and-Execute Agent** | Two-stage: planner generates a full task plan; executor works through each step |
| 38 | **Self-Ask with Search Agent** | Decompose complex questions into sub-questions answered by intermediate search calls |
| 39 | **Conversational Agent with Memory** | Agent retains conversation context across turns using integrated memory modules |
| 40 | **Multi-Tool Agents** | Equip agents with diverse tool sets (search, code executor, browser, DB) in a single loop |
| 41 | **AgentExecutor with Error Handling** | Configure stop criteria, max iterations, and parse error recovery for robust production agents |
| 42 | **Tool Choice Forcing** | Constrain agent to always call a specific tool or any tool (vs. auto-decide) |

---

### 5. Callbacks & Observability

| # | Technique | Description |
|---|-----------|-------------|
| 43 | **Custom CallbackHandler** | Subclass `BaseCallbackHandler` to hook into LLM, chain, tool, and retrieval lifecycle events |
| 44 | **Async Callbacks** | Non-blocking observability with `AsyncCallbackHandler` for high-throughput pipelines |
| 45 | **LangSmith Tracing** | Full run-tree capture with inputs, outputs, latencies, and costs sent to LangSmith |
| 46 | **Token Usage Tracking** | Aggregate prompt and completion token counts per run via `get_openai_callback()` |
| 47 | **Streaming Callbacks (`on_llm_new_token`)** | Process individual tokens as they stream out, enabling real-time token-level UIs |
| 48 | **Metadata & Tags on Runs** | Attach custom tags and metadata to runs for filtering and grouping in LangSmith |

---

### 6. Prompt Engineering

| # | Technique | Description |
|---|-----------|-------------|
| 49 | **ChatPromptTemplate with Roles** | Compose system, human, and AI role messages with dynamic variable placeholders |
| 50 | **FewShotPromptTemplate** | Dynamically select and format few-shot examples using example selectors |
| 51 | **SemanticSimilarityExampleSelector** | Retrieve the most semantically similar examples from a pool for few-shot prompts |
| 52 | **MaxMarginalRelevanceExampleSelector** | Balance relevance and diversity when selecting few-shot examples (MMR algorithm) |
| 53 | **PartialPromptTemplate** | Pre-bind some variables of a prompt template, leaving others for runtime injection |
| 54 | **PipelinePromptTemplate** | Compose multiple sub-prompts into a final prompt via a pipeline of string injections |
| 55 | **MessagesPlaceholder** | Reserve a slot in a chat prompt to inject a list of `BaseMessage` objects at runtime |
| 56 | **Prompt Versioning with LangSmith Hub** | Push/pull and version-control prompts via the LangSmith prompt hub |

---

### 7. Document Loaders & Transformers

| # | Technique | Description |
|---|-----------|-------------|
| 57 | **Custom Document Loaders** | Implement `BaseLoader` to load from any proprietary data source or API |
| 58 | **RecursiveCharacterTextSplitter** | Hierarchically split documents by paragraph → sentence → word for optimal chunk boundaries |
| 59 | **SemanticChunker** | Split documents at semantic breakpoints detected by embedding similarity drops |
| 60 | **HTMLHeaderTextSplitter** | Structure-aware splitting that preserves HTML header metadata in chunk metadata |
| 61 | **MarkdownHeaderTextSplitter** | Split Markdown by headers and propagate header hierarchy into chunk metadata |
| 62 | **CodeSplitter (Language-Aware)** | AST-aware splitting for source code that respects function and class boundaries |
| 63 | **Document Transformers** | Apply arbitrary transformations (translation, summarization, extraction) to loaded docs |

---

### 8. Output Parsers

| # | Technique | Description |
|---|-----------|-------------|
| 64 | **PydanticOutputParser** | Parse LLM output into a strongly-typed Pydantic model with automatic format instructions |
| 65 | **JsonOutputParser** | Stream and parse partial JSON output, supporting incremental rendering |
| 66 | **StructuredOutputParser** | Return multiple fields with custom schemas when native function calling isn't available |
| 67 | **OutputFixingParser** | Auto-retry parsing errors by sending the malformed output back to the LLM for self-correction |
| 68 | **RetryWithErrorOutputParser** | Re-prompt with the original input plus the parse error to repair bad outputs |
| 69 | **CommaSeparatedListOutputParser** | Parse a flat comma-delimited list from LLM output into a Python list |
| 70 | **DatetimeOutputParser** | Extract and parse ISO-format dates from natural language LLM responses |
| 71 | **EnumOutputParser** | Constrain LLM output to a predefined set of enum values |

---

### 9. Embeddings & Vector Stores

| # | Technique | Description |
|---|-----------|-------------|
| 72 | **Multi-Vector Retrieval** | Store multiple representations (summary, hypothetical question, chunk) per document |
| 73 | **Hybrid Search (Dense + Sparse)** | Combine BM25 keyword scores with dense vector scores for best-of-both retrieval |
| 74 | **Namespace / Collection Isolation** | Partition vector stores by tenant, topic, or access scope for multi-tenant RAG |
| 75 | **Incremental Indexing** | Detect and index only new or changed documents using document ID hashing |
| 76 | **Metadata Filtering at Query Time** | Apply structured filters (e.g., `{"source": "docs", "year": 2024}`) during vector search |
| 77 | **Matryoshka Embeddings (MRL)** | Use truncatable embeddings to trade accuracy for speed at query time |

---

### 10. Streaming

| # | Technique | Description |
|---|-----------|-------------|
| 78 | **`.stream()` on LCEL Chains** | Propagate streaming tokens through the full chain, not just the LLM layer |
| 79 | **`.astream()` Async Streaming** | Non-blocking async streaming for FastAPI / async web frameworks |
| 80 | **`.astream_events()` Event Streaming** | Fine-grained event stream exposing start/end/chunk events for every chain component |
| 81 | **Server-Sent Events (SSE) Integration** | Bridge LangChain streaming to browser clients via SSE endpoints |

---

## 🕸️ LangGraph — Advanced Techniques

### 1. Graph Architecture

| # | Technique | Description |
|---|-----------|-------------|
| 82 | **StateGraph** | Define a stateful directed graph where nodes read/write to a shared typed state schema |
| 83 | **MessageGraph** | Specialized graph where state is a list of messages — ideal for chat agent loops |
| 84 | **Node Definition (Functions & Runnables)** | Register async functions or LCEL runnables as graph nodes with `add_node()` |
| 85 | **Typed State with TypedDict / Pydantic** | Enforce state schema at compile time using Python TypedDict or Pydantic BaseModel |
| 86 | **Annotated State Fields with Reducers** | Use `Annotated[type, reducer_fn]` to control how node outputs are merged into shared state |
| 87 | **Subgraphs** | Nest an entire compiled graph as a single node inside a parent graph for modularity |
| 88 | **Graph Compilation (`compile()`)** | Compile a graph to validate structure, attach checkpointers, and produce a `CompiledGraph` |
| 89 | **Input / Output Schema Narrowing** | Specify `input` and `output` schemas to expose only a subset of state to callers |

---

### 2. State Management

| # | Technique | Description |
|---|-----------|-------------|
| 90 | **Reducer Functions (add, replace, custom)** | Control merge semantics per field: append to lists, replace scalars, or apply custom logic |
| 91 | **Operator.add Reducer** | Built-in reducer that appends new values to a list field (e.g., message accumulation) |
| 92 | **Private State Channels** | Mark state fields as private so they are not exposed in subgraph inputs/outputs |
| 93 | **State Snapshots** | Capture and restore exact state at any point in time for debugging or branching |
| 94 | **Shared State Across Subgraphs** | Pass and merge state between parent and child graphs during subgraph invocations |

---

### 3. Control Flow

| # | Technique | Description |
|---|-----------|-------------|
| 95 | **Conditional Edges** | Route to different nodes based on a function that inspects current state (dynamic branching) |
| 96 | **Static Edges** | Deterministic unconditional edges that always proceed to the next node |
| 97 | **`END` Terminus Node** | Terminate graph execution by routing to the special built-in `END` node |
| 98 | **`START` Entry Node** | Define the entry point of graph execution via `START` node constant |
| 99 | **Fan-Out / Fan-In Patterns** | Dispatch from one node to many in parallel, then merge results into a single aggregator node |
| 100 | **Send API (Dynamic Edge Targets)** | Emit `Send(node, state)` objects from conditional edges to invoke a node with custom state |
| 101 | **Map-Reduce over State Lists** | Use `Send` to scatter list items across parallel node invocations and reduce results |
| 102 | **Cyclic Graphs (Agent Loops)** | Create deliberate cycles (agent → tool → agent) to implement iterative reasoning loops |
| 103 | **Graph Recursion Limit** | Set `recursion_limit` at invocation time to prevent infinite loops in cyclic graphs |

---

### 4. Persistence & Checkpointing

| # | Technique | Description |
|---|-----------|-------------|
| 104 | **MemorySaver Checkpointer** | In-process memory checkpointer for development and testing (non-persistent) |
| 105 | **SqliteSaver Checkpointer** | Persist checkpoints to a local SQLite database for lightweight production use |
| 106 | **PostgresSaver Checkpointer** | Production-grade checkpointing to PostgreSQL for scalable, multi-process deployments |
| 107 | **Thread-Level Persistence** | Isolate conversation state per user/thread using `config={"configurable": {"thread_id": "..."}}` |
| 108 | **Time-Travel Debugging** | Replay or branch execution from any past checkpoint to inspect or fix agent decisions |
| 109 | **Checkpoint Namespaces** | Scope checkpoints within threads using namespaces for nested or multi-session scenarios |
| 110 | **Cross-Thread State (Memory Store)** | Share state across threads using a persistent `BaseStore` for user-level long-term memory |
| 111 | **Custom Checkpointer Implementation** | Implement `BaseCheckpointSaver` to persist state to any backend (Redis, DynamoDB, etc.) |

---

### 5. Multi-Agent Systems

| # | Technique | Description |
|---|-----------|-------------|
| 112 | **Supervisor Agent Pattern** | A controller LLM routes tasks to specialized worker agents and aggregates their results |
| 113 | **Hierarchical Agent Teams** | Nest supervisor graphs — supervisors manage sub-supervisors, which manage workers |
| 114 | **Agent-as-Node Pattern** | Encapsulate a complete LangChain agent (with tools) as a single node in a LangGraph |
| 115 | **Handoff / Transfer Tools** | Define tools that agents use to explicitly transfer control to another agent by name |
| 116 | **Swarm Architecture** | Peer-to-peer agent network where any agent can hand off to any other agent dynamically |
| 117 | **Network of Agents** | Broadcast tasks to all agents and aggregate the best response using a judge agent |
| 118 | **Shared Message Bus** | Route messages through a shared state channel that all agents read from and write to |
| 119 | **Specialized Worker Agents** | Assign narrow domain expertise (search, code, math, browser) to dedicated agent nodes |

---

### 6. Human-in-the-Loop

| # | Technique | Description |
|---|-----------|-------------|
| 120 | **`interrupt()` Function** | Pause graph execution mid-node and surface a value to the caller for human review |
| 121 | **`interrupt_before` on Edges** | Declare that execution should pause before a specified node without modifying node code |
| 122 | **`interrupt_after` on Edges** | Pause execution immediately after a node completes, before the next node begins |
| 123 | **Resume with `Command(resume=...)`** | Provide human feedback to the paused `interrupt()` call and resume graph execution |
| 124 | **Tool Call Approval Workflow** | Interrupt before tool execution to let a human approve, edit, or reject the pending call |
| 125 | **State Editing at Breakpoints** | Modify any state field while paused at an interrupt before resuming execution |
| 126 | **Multi-Turn Human Feedback Loops** | Chain multiple interrupts to gather iterative human input throughout a long-running task |
| 127 | **Streaming State to a Review UI** | Stream partial state to a frontend dashboard so humans can monitor and intervene live |

---

### 7. Streaming & Observability

| # | Technique | Description |
|---|-----------|-------------|
| 128 | **`.stream()` Mode — Values** | Stream the full state snapshot after every node completes |
| 129 | **`.stream()` Mode — Updates** | Stream only the state delta (diff) produced by each node, not the full state |
| 130 | **`.stream()` Mode — Messages** | Stream individual LLM tokens as they are generated from within any node |
| 131 | **`.stream()` Mode — Debug** | Emit detailed internal debug events (checkpoint saves, task scheduling) for deep inspection |
| 132 | **`.astream_events()` v2** | Fine-grained async event stream with run_id, tags, metadata per LangChain component |
| 133 | **Custom Stream Modes** | Combine multiple stream modes (e.g., `["values", "messages"]`) in a single invocation |
| 134 | **LangSmith Auto-Tracing** | Automatically capture full graph run trees with node-level timing when tracing is enabled |
| 135 | **Metadata & Tags on Graph Nodes** | Attach tags and metadata to individual nodes for fine-grained filtering in LangSmith |

---

### 8. Deployment & Production

| # | Technique | Description |
|---|-----------|-------------|
| 136 | **LangGraph Platform (Cloud / Self-Hosted)** | Deploy graphs as managed APIs with built-in persistence, streaming, and cron scheduling |
| 137 | **LangGraph CLI (`langgraph dev`)** | Local development server with hot-reload and built-in LangGraph Studio UI |
| 138 | **LangGraph Studio** | Visual graph debugger — inspect state, replay runs, and edit state at breakpoints via GUI |
| 139 | **Background Task Runs** | Dispatch long-running graph executions as background jobs and poll for completion |
| 140 | **Cron-Scheduled Graphs** | Trigger graph runs on a schedule (e.g., daily digest, periodic data refresh) |
| 141 | **Webhook Callbacks on Run Completion** | Post results to a URL when a background graph run finishes |
| 142 | **Double-Texting Handling** | Configure `enqueue`, `interrupt`, `rollback`, or `reject` policies for concurrent user messages |
| 143 | **Custom Auth & Middleware** | Plug in authentication handlers and middleware at the LangGraph Platform layer |
| 144 | **Horizontal Scaling with Shared Postgres** | Run multiple graph workers against a shared PostgreSQL checkpointer for stateful scale-out |
| 145 | **Assistants API** | Manage multiple named configurations (assistants) of the same graph for A/B testing |

---

## 🔀 Integration Patterns

| # | Pattern | Involves | Description |
|---|---------|----------|-------------|
| 146 | **Agentic RAG** | LangChain RAG + LangGraph | Agent decides when and how many times to retrieve, adaptively refining queries |
| 147 | **Tool-Calling Agent Loop** | LangChain Tools + LangGraph Cycles | Persistent cyclic graph wrapping a tool-using agent with stateful multi-turn memory |
| 148 | **Corrective RAG (CRAG)** | LangGraph + LangChain Retriever | Grade retrieved docs; fallback to web search if quality is below threshold |
| 149 | **Self-RAG** | LangGraph + LangChain | Model decides when to retrieve and critiques its own generation for faithfulness |
| 150 | **Essay Writer / Long-Form Generator** | LangGraph + LangChain | Multi-step graph: plan → research → draft → reflect → revise loop |
| 151 | **Code Generation + Execution Loop** | LangGraph + LangChain Code Tool | Generate code → execute in sandbox → inspect error → regenerate in a feedback cycle |
| 152 | **Multi-Agent Research Pipeline** | LangGraph Supervisor + LangChain Tools | Supervisor delegates to search, summarize, and critique agents in a research workflow |
| 153 | **Streaming Chatbot with Memory** | LangGraph Persistence + LangChain Chat | Checkpointed thread per user, streaming LLM tokens to the frontend via SSE |

---

## 📚 Key Resources

| Resource | URL |
|----------|-----|
| LangChain Docs | https://python.langchain.com/docs |
| LangGraph Docs | https://langchain-ai.github.io/langgraph |
| LangSmith | https://smith.langchain.com |
| LangGraph Platform | https://www.langchain.com/langgraph-platform |
| LangChain GitHub | https://github.com/langchain-ai/langchain |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |
| LangChain Hub (Prompts) | https://smith.langchain.com/hub |

---

> **Playlist Total: 153 Techniques** — spanning chains, RAG, memory, agents, graph architecture, state management, control flow, persistence, multi-agent systems, human-in-the-loop, streaming, and deployment patterns.
