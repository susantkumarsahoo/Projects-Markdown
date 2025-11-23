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