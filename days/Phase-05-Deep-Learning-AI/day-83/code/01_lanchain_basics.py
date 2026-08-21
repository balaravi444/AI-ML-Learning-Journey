"""
Day 83 — LangChain Basics
Topic: Prompts, Chains, Memory
Date: 09 August 2026
Author: Bala Ravi

LangChain connects LLMs to real applications.
Building blocks for MemoryOS!
"""
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI)
    from langchain_core.prompts import (
        ChatPromptTemplate,
        PromptTemplate)
    from langchain_core.output_parsers import (
        StrOutputParser,
        JsonOutputParser)
    from langchain_core.messages import (
        HumanMessage, SystemMessage)
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not installed.")
    print("    Run: pip install langchain "
          "langchain-google-genai\n")

# Set API key
GEMINI_API_KEY = os.environ.get(
    'GEMINI_API_KEY', 'your-api-key-here')


def basic_llm_demo() -> None:
    """Show basic LLM usage with LangChain."""
    print("=== Basic LLM with LangChain ===\n")

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available.")
        print("\nWhat this would do:\n")

        code = '''
# Without LangChain (manual):
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("What is RAG?")
print(response.text)

# With LangChain (clean interface):
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY)

response = llm.invoke([
    HumanMessage(content="What is RAG?")
])
print(response.content)
# "RAG stands for Retrieval Augmented Generation..."
        '''
        print(code)
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1)

    response = llm.invoke([
        SystemMessage(
            content="You are a helpful AI assistant. "
                    "Answer concisely in 2-3 sentences."),
        HumanMessage(
            content="What is RAG in AI?")
    ])

    print(f"Q: What is RAG in AI?")
    print(f"A: {response.content}\n")


def prompt_template_demo() -> None:
    """Show PromptTemplate usage."""
    print("=== Prompt Templates ===\n")

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available.\n")
        print("PromptTemplate concept:\n")

        template_code = '''
# PromptTemplate: parameterized prompts!

template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in {domain}. "
     "Answer in {style} style."),
    ("human", "{question}")
])

# Fill in variables
prompt = template.format_messages(
    domain="machine learning",
    style="simple beginner-friendly",
    question="What is overfitting?")

# Chain: template → LLM → output
chain = template | llm | StrOutputParser()
response = chain.invoke({
    "domain": "machine learning",
    "style": "simple",
    "question": "What is overfitting?"
})
# "Overfitting is when your model memorizes
#  training data instead of learning patterns..."
        '''
        print(template_code)
        return

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert in {domain}. "
         "Explain concepts simply in 2-3 sentences."),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1)

    chain = template | llm | StrOutputParser()

    questions = [
        {"domain": "machine learning",
         "question": "What is overfitting?"},
        {"domain": "neural networks",
         "question": "What is dropout?"}
    ]

    for q in questions:
        response = chain.invoke(q)
        print(f"Domain: {q['domain']}")
        print(f"Q: {q['question']}")
        print(f"A: {response}\n")


def sequential_chain_demo() -> None:
    """Show chaining multiple LLM calls."""
    print("=== Sequential Chains ===\n")
    print("Output of Step 1 → Input of Step 2\n")

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available.\n")
        print("Sequential chain concept:\n")

        code = '''
# Step 1: Generate bug analysis
step1 = ChatPromptTemplate.from_messages([
    ("human",
     "Analyze this bug report in one sentence: {bug}")
]) | llm | StrOutputParser()

# Step 2: Generate fix suggestion
step2 = ChatPromptTemplate.from_messages([
    ("human",
     "Suggest a fix for: {analysis}")
]) | llm | StrOutputParser()

# Chain steps together!
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"bug": RunnablePassthrough()}
    | step1
    | (lambda x: {"analysis": x})
    | step2)

result = chain.invoke(
    "Production server crashed at 3am")
# Step 1: "Critical production failure at night"
# Step 2: "Set up on-call rotation + monitoring"
        '''
        print(code)
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1)

    # Bug analyzer → priority suggester chain
    analyze_prompt = ChatPromptTemplate.from_messages([
        ("human",
         "In one sentence, analyze severity of: {bug}")
    ])
    priority_prompt = ChatPromptTemplate.from_messages([
        ("human",
         "Based on: '{analysis}'\n"
         "Assign priority: Critical/High/Medium/Low "
         "and explain why in one sentence.")
    ])

    analyze_chain = (
        analyze_prompt | llm | StrOutputParser())

    def run_chain(bug_report: str) -> None:
        analysis = analyze_chain.invoke(
            {"bug": bug_report})
        priority_chain = (
            priority_prompt | llm | StrOutputParser())
        priority = priority_chain.invoke(
            {"analysis": analysis})

        print(f"Bug: {bug_report[:50]}")
        print(f"Analysis: {analysis}")
        print(f"Priority: {priority}\n")

    bugs = [
        "Production DB crashed, all users locked out",
        "Typo in footer copyright year"
    ]

    for bug in bugs:
        run_chain(bug)


def memory_demo() -> None:
    """Show conversation memory."""
    print("=== Conversation Memory ===\n")

    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available.\n")
        print("Memory concept:\n")

        conversation = [
            ("Human", "My name is Bala Ravi"),
            ("AI", "Hello Bala Ravi! How can I help?"),
            ("Human", "What is my name?"),
            ("AI", "Your name is Bala Ravi, "
                   "as you told me earlier."),
            ("Human", "What am I building?"),
            ("AI", "You haven't told me what "
                   "you're building yet.")
        ]

        for speaker, text in conversation:
            print(f"  {speaker}: {text}")

        print("\n💡 With memory: AI remembers 'Bala Ravi'!")
        print("   Without memory: AI forgets every turn!")
        print("   MemoryOS NEEDS this! 🔥")
        return

    from langchain.memory import (
        ConversationBufferMemory)
    from langchain.chains import ConversationChain

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1)

    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, memory=memory, verbose=False)

    turns = [
        "My name is Bala Ravi and I'm building MemoryOS",
        "What is my name?",
        "What am I building?",
        "Describe what MemoryOS does in one sentence."
    ]

    for turn in turns:
        response = conversation.predict(
            input=turn)
        print(f"Human: {turn}")
        print(f"AI: {response[:100]}...\n")


if __name__ == "__main__":
    basic_llm_demo()
    prompt_template_demo()
    sequential_chain_demo()
    memory_demo()
