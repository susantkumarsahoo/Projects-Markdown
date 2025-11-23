# LangGraph Techniques Reference

## 1. Core Concepts
**Graph Fundamentals**: nodes, edges, state_graph, workflow_graph, llm_graph, conditional_edges, branching, control_flow, cycles, recursion, checkpointing, persistence, stream_events, interruptible_steps

## 2. State Management
**State Techniques**: state, state_schema, typed_state, pydantic_state, annotated_state, readonly_state, reducer_functions, state_transitions, atomic_state_updates

## 3. Node Types
**Node Functions**: fn_node, tool_node, llm_node, model_node, retriever_node, router_node, map_node, reduce_node, parallel_node, human_node, dynamic_node, conditional_node

## 4. Execution Engine
**Runnables**: runnable, runnable_lambda, runnable_binding, runnable_parallel, runnable_sequence, async_runnables, streaming_runnables, invoke, ainvoke, batch, abatch

## 5. Control Flow
**Flow Control**: if_else_edges, conditional_edges, branching_logic, loops, cycles, retry_edges, fallback_edges, guardrails, validation_nodes

## 6. Memory & Persistence
**Storage**: memory, durable_memory, checkpointing, sqlite_checkpoint, redis_checkpoint, in_memory_checkpoint, message_history, thread_state, conversation_state, snapshotting, resumable_graph

## 7. Tools Integration
**Tool Techniques**: tools, tool_execution, tool_binding, tool_validation, structured_tools, mcp_tools, agent_tools, tool_router

## 8. Agents
**Agent Types**: react_agent, tool_agent, chat_agent, workflow_agent, agent_state, agent_loop, interruptible_agent, agent_with_memory, agent_action_nodes, agent_executor

## 9. Multi-Agent Systems
**Multi-Agent**: multi_agent_graph, agent_to_agent_edges, coordinator_agent, worker_agent, supervisor_agent, routing_agent, delegation, group_chat_graph

## 10. Streaming
**Stream Types**: event_streaming, step_streaming, token_streaming, stream_updates, stream_graph_execution, observable_events

## 11. Error Handling
**Failure Management**: error_edges, fallback_edges, retries, exception_handlers, safe_nodes, validation_nodes, guardrail_nodes

## 12. LangChain Integration
**Integrations**: langchain_runnables, langchain_tools, retrievers, embeddings, vectorstores, LCEL_with_graph, chains_inside_graph

## 13. Deployment
**Serving**: langgraph_server, async_api, batch_api, websocket_streaming, rest_api, agent_server, cloud_deploy, langgraph_cloud

## 14. Monitoring
**Observability**: graph_debugger, step_inspector, graph_visualizer, artifact_view, logs, tracing, telemetry

## 15. Advanced Features
**Enterprise**: self_healing_flows, resumable_agents, human_in_the_loop, approval_edges, audit_logs, sandbox_execution, role_based_routes

---

## Quick Reference (All Techniques)
nodes • edges • state • state_schema • fn_node • llm_node • tool_node • control_flow • conditional_edges • branching • loops • cycles • persistence • checkpointing • durable_memory • tools • agents • multi_agents • runnables • streaming • retries • guardrails • server • deployment • tracing • debugging • visualizer • cloud • human_in_the_loop • memory • asynch • batch • validation • error_handling • fallback • router • parallel • reduce • map • react_agent • supervisor • coordinator • worker • delegation • embeddings • vectorstores • LCEL • telemetry • audit_logs • sandbox • resumable • snapshot • thread_state • conversation_state • observable_events • token_streaming


# LangGraph Complete Reference

## 1. Core Concepts

**What is LangGraph?**
- Stateful graph framework for complex LLM workflows
- Supports cyclic graphs (loops + conditionals)
- Built on LangChain with native persistence
- Enables agent workflows with checkpointing

**Graph Types**
- `StateGraph` - Main graph with typed state
- `MessageGraph` - Specialized for chat
- `WorkflowGraph` - Simple stateless workflow

**Key Components**
- **Nodes**: Functions that process state
- **Edges**: Connections (normal/conditional/dynamic)
- **State**: Shared typed data structure
- **Checkpoints**: Persistence save points

---

## 2. State Management

**State Schema**
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer
    context: str
    counter: int
```

**State Types**
- `TypedDict` - Python typed dictionary
- `Pydantic` - Validated models
- `Annotated` - With reducer functions

**Reducers**
- `add_messages` - Append to message list
- Custom: `def merge(old, new) -> merged`

**Operations**
- Read: Access in nodes
- Update: Return new values (atomic per node)
- Readonly: Mark immutable fields

---

## 3. Building Graphs

**Basic Structure**
```python
from langgraph.graph import StateGraph, END

# 1. Define state
class State(TypedDict):
    input: str
    output: str

# 2. Create graph
graph = StateGraph(State)

# 3. Add nodes
graph.add_node("process", process_fn)

# 4. Add edges
graph.add_edge("process", END)

# 5. Set entry
graph.set_entry_point("process")

# 6. Compile
app = graph.compile()
```

**Node Function**
```python
def my_node(state: State) -> dict:
    return {"output": state["input"].upper()}
```

---

## 4. Edges & Control Flow

**Edge Types**

**Normal Edge**
```python
graph.add_edge("node_a", "node_b")
```

**Conditional Edge**
```python
def router(state: State) -> str:
    return "high" if state["score"] > 0.8 else "low"

graph.add_conditional_edges(
    "classifier",
    router,
    {"high": "approve", "low": "reject"}
)
```

**Dynamic Edge**
```python
def dynamic_route(state: State) -> str:
    return state["next_node"]  # Runtime decision
```

**Patterns**
- Branching, loops, parallel execution
- Conditional routing, retry logic

---

## 5. Node Types

**Function Node**
```python
def process(state: State) -> dict:
    return {"result": compute(state["input"])}
```

**LLM Node**
```python
def llm_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

**Tool Node**
```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([search_tool, calc_tool])
```

**Router Node**
```python
def router(state: State) -> str:
    return "search" if "?" in state["query"] else "answer"
```

---

## 6. Checkpointing & Persistence

**Memory Savers**
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# SQLite
with SqliteSaver.from_conn_string("db.sqlite") as saver:
    app = graph.compile(checkpointer=saver)
```

**Usage**
```python
config = {"configurable": {"thread_id": "user_1"}}

# First call - saves checkpoint
result = app.invoke({"input": "Hi"}, config)

# Resume from checkpoint
result = app.invoke({"input": "Continue"}, config)
```

**Benefits**
- Resume workflows, time travel states
- Branching execution paths
- Debugging + inspection

---

## 7. Agents

**Prebuilt ReAct Agent**
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)
```

**Custom Agent Loop**
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def tool_node(state):
    tool_calls = state["messages"][-1].tool_calls
    results = [tool.invoke(call) for call in tool_calls]
    return {"messages": results}

def should_continue(state) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
```

**Features**
- Tool calling, reasoning loops
- Memory, interruptible execution

---

## 8. Multi-Agent Systems

**Supervisor Pattern**
```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_worker: str

def supervisor(state):
    response = supervisor_llm.invoke(state["messages"])
    return {"next_worker": parse_worker(response)}

# Build supervisor graph
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_worker"],
    {"researcher": "researcher", "writer": "writer"}
)
```

**Patterns**
- Supervisor → Workers
- Agent → Agent communication
- Shared state, message passing

---

## 9. Streaming

**Methods**
```python
# Stream outputs
for chunk in app.stream({"input": "Hi"}):
    print(chunk)

# Stream events (detailed)
for event in app.stream_events({"input": "Hi"}):
    print(event)

# Async streaming
async for chunk in app.astream({"input": "Hi"}):
    print(chunk)
```

**Types**
- Node outputs, LLM tokens
- Execution events, state updates

---

## 10. Human-in-the-Loop

**Interrupt Points**
```python
# Interrupt before node
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approval"]
)

# Execute until interrupt
result = app.invoke({"input": "data"}, config)

# Resume after approval
result = app.invoke(None, config)
```

**Update State During Interrupt**
```python
# Get state
state = app.get_state(config)

# Modify
app.update_state(config, {"approved": True})

# Resume
result = app.invoke(None, config)
```

---

## 11. Loops & Cycles

**Simple Loop**
```python
def should_continue(state: State) -> str:
    if state["counter"] < 5:
        return "process"  # Loop back
    return END

graph.add_conditional_edges(
    "process",
    should_continue,
    {"process": "process", END: END}
)
```

**Retry Logic**
```python
def retry_route(state: State) -> str:
    if state["attempts"] < 3 and state["error"]:
        return "retry"
    return "next"
```

**Recursion Limit**
```python
app = graph.compile(recursion_limit=100)  # Default: 25
```

---

## 12. Error Handling

**Try-Catch in Nodes**
```python
def safe_node(state: State) -> dict:
    try:
        result = risky_op(state["input"])
        return {"output": result, "error": None}
    except Exception as e:
        return {"error": str(e)}
```

**Fallback Edges**
```python
def error_route(state: State) -> str:
    return "fallback" if state.get("error") else "success"

graph.add_conditional_edges(
    "risky",
    error_route,
    {"fallback": "fallback_node", "success": "next"}
)
```

---

## 13. LangChain Integration

**LCEL in Nodes**
```python
from langchain_core.prompts import ChatPromptTemplate

chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | llm
    | StrOutputParser()
)

def node(state: State) -> dict:
    return {"summary": chain.invoke({"text": state["content"]})}
```

**Retrievers**
```python
def rag_node(state: State) -> dict:
    docs = retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in docs])
    return {"context": context}
```

---

## 14. Deployment

**LangGraph Server**
```python
from langgraph.server import create_app

app = create_app(compiled_graph)
# Run: uvicorn server:app
```

**API Endpoints**
- `POST /invoke` - Single call
- `POST /stream` - Streaming
- `POST /batch` - Batch processing
- `GET /state/{thread_id}` - Get state
- `POST /state/{thread_id}` - Update state

---

## 15. Debugging & Monitoring

**Visualize**
```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

**Inspect State**
```python
state = app.get_state(config)
print(state.values)  # Current state
print(state.next)    # Next nodes
```

**Tracing (LangSmith)**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "key"
```

**Step-by-Step**
```python
for step in app.stream(input, config, stream_mode="debug"):
    print(f"Node: {step['node']}, State: {step['state']}")
```

---

## 16. Advanced Patterns

**Map-Reduce**
```python
def map_node(state):
    results = [process(item) for item in state["items"]]
    return {"mapped": results}

def reduce_node(state):
    return {"final": combine(state["mapped"])}
```

**Parallel Execution**
```python
# Multiple nodes run simultaneously
graph.add_edge("start", "node_a")
graph.add_edge("start", "node_b")
graph.add_edge("start", "node_c")

# Join results
graph.add_edge("node_a", "join")
graph.add_edge("node_b", "join")
graph.add_edge("node_c", "join")
```

**Subgraphs**
```python
# Create reusable subgraph
subgraph = StateGraph(SubState)
compiled_sub = subgraph.compile()

# Use in main graph
def use_subgraph(state):
    result = compiled_sub.invoke({"input": state["data"]})
    return {"output": result}
```

---

## 17. Common Workflows

**Chat App**
```python
from langgraph.graph import MessageGraph

def chatbot(messages):
    return [llm.invoke(messages)]

graph = MessageGraph()
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
app = graph.compile(checkpointer=MemorySaver())
```

**RAG**
```python
class RAGState(TypedDict):
    question: str
    documents: list
    answer: str

def retrieve(state):
    return {"documents": retriever.invoke(state["question"])}

def generate(state):
    context = "\n".join([d.page_content for d in state["documents"]])
    answer = llm.invoke(f"Context: {context}\n\nQ: {state['question']}")
    return {"answer": answer}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge("retrieve", "generate")
```

---

## 18. Best Practices

**State Design**
- Keep flat and simple
- Use TypedDict/Pydantic
- Use reducers for lists/dicts
- Avoid deep nesting

**Node Design**
- One responsibility per node
- Return partial updates only
- Pure functions when possible
- Handle errors gracefully

**Graph Structure**
- Start simple, add complexity
- Use conditional edges for routing
- Add checkpointing for long workflows
- Visualize during development

**Performance**
- Use async for I/O operations
- Batch when possible
- Cache expensive operations
- Set appropriate recursion limits

---

## 19. LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Structure | Linear chains | Cyclic graphs |
| State | Passed through | Persistent object |
| Loops | Not native | Built-in |
| Conditional | Limited | Full routing |
| Persistence | Memory modules | Checkpointing |
| Human-in-loop | Manual | Native interrupts |
| Best for | Simple flows | Complex agents |

---

## 20. Essential Checklist

- [ ] Define state schema (TypedDict)
- [ ] Add checkpointer for persistence
- [ ] Use conditional edges for routing
- [ ] Implement error handling
- [ ] Enable streaming for UX
- [ ] Add human-in-the-loop where needed
- [ ] Set recursion_limit appropriately
- [ ] Visualize graph during dev
- [ ] Use thread_id for multi-user
- [ ] Test with interrupt_before/after

---

## Production Example

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int

def agent(state):
    llm = ChatOpenAI(model="gpt-4").bind_tools(tools)
    response = llm.invoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}

tool_node = ToolNode(tools)

def should_continue(state) -> str:
    if state["iterations"] >= 10:
        return END
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

with SqliteSaver.from_conn_string("agent.db") as saver:
    app = graph.compile(checkpointer=saver, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user_1"}}
result = app.invoke(
    {"messages": [("user", "Research AI trends")], "iterations": 0},
    config
)
```

---

## Quick Reference Tags

`nodes` `edges` `state` `StateGraph` `MessageGraph` `add_node` `add_edge` `add_conditional_edges` `set_entry_point` `compile` `invoke` `stream` `astream` `checkpointer` `MemorySaver` `SqliteSaver` `thread_id` `interrupt_before` `interrupt_after` `END` `TypedDict` `add_messages` `ToolNode` `create_react_agent` `get_state` `update_state` `recursion_limit` `draw_mermaid_png` `LCEL` `RAG` `multi_agent` `supervisor` `human_in_the_loop` `streaming` `async` `error_handling` `retry` `fallback` `parallel` `subgraphs` `map_reduce` `LangSmith` `tracing` `deployment` `API` `server`

---

**Remember**: LangGraph = LangChain + Graphs + Persistence + Cycles + Agents

