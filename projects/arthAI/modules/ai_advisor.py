"""
ArthAI — AI Chatbot Module
LLM-powered financial advisor using Gemini API.

Author: Bala Ravi
Date: 19 June 2026
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
You are ArthAI — an elite, highly sophisticated Wealth Manager and Financial Advisor built specifically for Indians. You provide premium, expert-level financial guidance.

RULES YOU MUST FOLLOW:
1. Always answer with a professional, authoritative, yet approachable tone, suitable for high-net-worth individuals or aspiring individuals.
2. Use Indian Rupee (₹) symbol for all amounts.
3. Strategically reference Indian financial instruments (PPF, ELSS, NPS, FD, Mutual Funds, EPF, Sukanya Samriddhi) with deep insight.
4. Provide structured, easy-to-read responses using **Markdown formatting** (bolding key terms, using bullet points).
5. NEVER give specific stock buy/sell recommendations.
6. End your response subtly with a disclaimer: *Note: This is an AI-generated strategy. Please consult a SEBI-registered advisor before executing.*
7. Keep responses dense with value but concise — under 300 words.
8. If asked about non-financial topics, politely pivot back to wealth creation.

Your personality: Like a seasoned, elite wealth manager who operates out of a high-end Mumbai financial district. You are sharp, analytical, and highly structured.
"""


def build_user_context(user_data: dict) -> str:
    """
    Build context string from user's financial profile.
    Simplified RAG — injecting relevant data!

    Args:
        user_data: Dictionary with user financial info

    Returns:
        Formatted context string
    """
    if not user_data:
        return ""

    context_parts = ["User's Financial Profile:"]

    if user_data.get('age'):
        context_parts.append(f"- Age: {user_data['age']}")
    if user_data.get('monthly_income'):
        context_parts.append(
            f"- Monthly Income: ₹{user_data['monthly_income']:,}")
    if user_data.get('monthly_savings'):
        context_parts.append(
            f"- Monthly Savings: ₹{user_data['monthly_savings']:,}")
    if user_data.get('existing_loans'):
        context_parts.append(
            f"- Existing Loan EMIs: ₹{user_data['existing_loans']:,}")
    if user_data.get('risk_appetite'):
        context_parts.append(
            f"- Risk Appetite: {user_data['risk_appetite']}")

    return "\n".join(context_parts) if len(context_parts) > 1 else ""


def build_prompt(
        question: str,
        user_data: dict = None,
        chat_history: list[dict] = None) -> str:
    """
    Build complete prompt with system instructions,
    user context, and conversation history.

    Args:
        question: User's current question
        user_data: User's financial profile
        chat_history: Previous conversation turns

    Returns:
        Complete formatted prompt
    """
    parts = [SYSTEM_PROMPT]

    context = build_user_context(user_data) if user_data else ""
    if context:
        parts.append(f"\n{context}\n")

    if chat_history:
        recent = chat_history[-3:]
        parts.append("\nRecent conversation:")
        for turn in recent:
            parts.append(f"User: {turn['user']}")
            parts.append(f"ArthAI: {turn['assistant']}")

    parts.append(f"\nCurrent question: {question}")
    parts.append("\nArthAI's response:")

    return "\n".join(parts)


def get_ai_response(
        question: str,
        user_data: dict = None,
        chat_history: list[dict] = None) -> str:
    """
    Get AI-powered financial advice response.

    Args:
        question: User's question
        user_data: Optional user financial profile
        chat_history: Optional conversation history

    Returns:
        AI generated response
    """
    if not GEMINI_API_KEY:
        return _get_fallback_response(question)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = build_prompt(question, user_data, chat_history)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.7,
            )
        )
        return response.text.strip()

    except Exception as e:
        print(f"AI Error: {e}")
        return _get_fallback_response(question)


def _get_fallback_response(question: str) -> str:
    """
    Rule-based fallback when AI is unavailable.
    Ensures app never completely breaks!

    Args:
        question: User's question

    Returns:
        Pre-written helpful response
    """
    question_lower = question.lower()

    if any(word in question_lower
           for word in ["sip", "mutual fund", "invest"]):
        return (
            "For investments, consider starting a SIP in "
            "diversified mutual funds. Even ₹500/month can "
            "grow significantly over time! Use our SIP "
            "Calculator above for exact numbers. "
            "Note: Consult a SEBI registered advisor for "
            "personalized advice."
        )

    if any(word in question_lower
           for word in ["tax", "80c", "save tax"]):
        return (
            "To save tax, consider ELSS mutual funds, PPF, "
            "or NPS under Section 80C (up to ₹1.5L deduction). "
            "Use our Tax Calculator above for your specific "
            "savings! Note: Consult a SEBI registered advisor "
            "for personalized advice."
        )

    if any(word in question_lower
           for word in ["loan", "emi", "debt"]):
        return (
            "For loans, always compare interest rates across "
            "banks. Consider prepaying when you have surplus "
            "funds to reduce total interest. Use our EMI "
            "Calculator above! Note: Consult a SEBI registered "
            "advisor for personalized advice."
        )

    if any(word in question_lower
           for word in ["retire", "retirement", "corpus"]):
        return (
            "For retirement, start early! Even ₹2,000/month "
            "SIP at age 25 can create ₹1 crore+ by age 55 "
            "at 12% returns. Use our Retirement Planner! "
            "Note: Consult a SEBI registered advisor for "
            "personalized advice."
        )

    if any(word in question_lower
           for word in ["budget", "salary", "spend"]):
        return (
            "Try the 50/30/20 rule: 50% for needs (rent, "
            "food), 30% for wants (entertainment), 20% for "
            "savings. Use our Budget Planner for a "
            "personalized breakdown! Note: Consult a SEBI "
            "registered advisor for personalized advice."
        )

    return (
        "I'm here to help with budgeting, investments, "
        "loans, tax planning, and retirement! Try asking "
        "something like 'How much should I save monthly?' "
        "Note: This is general guidance. Consult a SEBI "
        "registered advisor for personalized advice."
    )
