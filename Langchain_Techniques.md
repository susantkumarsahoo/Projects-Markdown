# LangChain
llms_chat_model, prompts,output_parsers, memory, chains, messages.py", rag_retrievers,
embedding, retrievers, runnables, callbacks, text_splitter, tools_toolkits, document_loader,
streaming.py", mcp_client,
        
# LangGraph

state, nodes, edges, checkpointer", persistence", workflows


# prompts

✅ 1. Prompt Templates (Core Prompt Types)

PromptTemplate, StringPromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate.from_template, ChatPromptTemplate.from_messages, TextPromptValue, ChatPromptValue, PromptValue

✅ 2. Few-Shot Prompting Techniques

FewShotPromptTemplate, FewShotChatMessagePromptTemplate, ExampleSelector, SemanticSimilarityExampleSelector, LengthBasedExampleSelector, NGramOverlapExampleSelector, SemanticSimilarityPromptTemplate

✅ 3. Dynamic & Flexible Prompting

MessagesPlaceholder, PartialPromptTemplate, Dynamic Prompt Templates, Runtime Prompt Injection, Template Variables, Multi-Lingual Prompt Support

✅ 4. Prompt Optimization & Compression

Prompt Compression, Context Condensation, Summarization-based Prompt Reduction, Adaptive Prompting, Multi-Prompt Routing

✅ 5. Structured Output & Validation

OutputParser, PydanticOutputParser, StructuredOutputParser, RetryOutputParser, Guardrails Integration, Schema-based Prompting

✅ 6. Advanced Prompt Routing Techniques

Multi-Prompt RouterChain, PromptSelector, Conditional Prompting, Router Chain + LLM Selection

✅ 7. Memory + Prompt Integration Techniques

ConversationBufferMemory Prompts, ConversationSummaryMemory Prompts, ConversationKGMemory Prompts, Chat History Injection via MessagesPlaceholder

✅ 8. Prompt Management & Storage

Prompt Serialization (load/save), JSON/YAML prompt storage, PromptLayer Integration, Prompt Registry, Prompt Versioning