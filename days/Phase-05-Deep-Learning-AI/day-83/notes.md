# Day 83 — LangChain Basics 🚀

**Date:** 09 August 2026
**Phase:** 5 — Deep Learning + AI
**Time Spent:** (add your hours)
**Resource Used:** [LangChain Docs](https://docs.langchain.com)

---

## 📚 Topics Covered

- What is LangChain and why it exists
- LLMs, Chat Models, Prompts
- PromptTemplates
- Chains — LLMChain, SequentialChain
- Memory — ConversationBufferMemory
- Output parsers
- Tool use basics
- Building with Gemini API

---

## 🔑 What is LangChain?
Without LangChain:
You call LLM API manually.
Parse response manually.
Manage conversation history manually.
Connect to tools manually.
Chain multiple prompts manually.
= Hundreds of lines of boilerplate!

With LangChain:
All of this in 10-20 lines!

LangChain = framework to build LLM applications
→ Manages prompts, chains, memory, tools
→ Connects any LLM to any data source
→ Standard interface for all LLMs! 🔥

---

## 🔑 Core Concepts
LLM / ChatModel → the AI brain

ChatOpenAI, ChatGoogleGenerativeAI, etc.

PromptTemplate → structured input

Variables + instructions → formatted prompt

Chain → sequence of operations

LLMChain: prompt → LLM → response
SequentialChain: output of one → input of next

Memory → conversation history

ConversationBufferMemory: stores all messages
ConversationSummaryMemory: summarizes old msgs
Output Parser → structure LLM output

StrOutputParser: raw string
JsonOutputParser: structured JSON
PydanticOutputParser: typed Python objects

---

## 🔑 LangChain Expression Language (LCEL)

```python
# Modern LangChain (LCEL):
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

# Chain: prompt | llm | parser
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "question": "What is RAG?"})
```

---

## 💎 Important Realizations

1. **LangChain = glue between LLM and everything**
   LLM is the brain.
   LangChain connects it to your database,
   your files, your APIs, your tools.
   Without LangChain → custom code for each!

2. **LCEL (pipe syntax) is elegant**
   prompt | llm | parser
   Left to right data flow.
   Easy to read, easy to modify!

3. **Memory is what makes chatbots useful**
   Without memory: "Who are you?" every message
   With memory: remembers entire conversation!
   MemoryOS needs memory to be useful!

4. **Gemini API is perfect for our projects**
   Free tier: 15 requests/minute
   Quality: matches GPT-3.5
   Already used in ArthAI! 🔥

---

## 🎯 Next Goal (Day 84)

- Vector Databases!
- ChromaDB — local, free, perfect for MemoryOS
- Store and retrieve embeddings at scale
- The memory store of MemoryOS!

---

*Day 83 complete — LangChain mastered! ⛓️🔥*


