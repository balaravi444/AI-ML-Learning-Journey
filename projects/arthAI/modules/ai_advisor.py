"""
ArthAI — AI Chatbot Module
LLM-powered financial advisor using Gemini API.

Author: Bala Ravi
Date: 19 June 2026
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

SYSTEM_PROMPT = """
You are ArthAI — a friendly, knowledgeable financial
advisor built specifically for Indians.

RULES YOU MUST FOLLOW:
1. Always answer in simple, jargon-free English
2. Use Indian Rupee (₹) symbol for all amounts
3. Reference Indian financial instruments when relevant
   (PPF, ELSS, NPS, FD, Mutual Funds, EPF, Sukanya Samriddhi)
4. NEVER give specific stock buy/sell recommendations
5. Always end with: "Note: This is general guidance.
   Consult a SEBI registered advisor for personalized advice."
6. Keep responses concise — under 200 words
7. Use simple real-number examples to explain concepts
8. Be warm and encouraging, never condescending
9. If asked about something outside personal finance,
   politely redirect to financial topics

Your personality: Like a knowledgeable older sibling
who explains money matters simply, without judgment.
"""


def build_user_context(user_data: dict) -> str:
    """
    Build context string from user's financial profile.
    This is simplified RAG — injecting relevant data!

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
            f"- Current Monthly Savings: ₹{user_data['monthly_savings']:,}")
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

    # Inject user context (simplified RAG!)
    context = build_user_context(user_data) if user_data else ""
    if context:
        parts.append(f"\n{context}\n")

    # Add limited recent history (last 3 exchanges only!)
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

    Raises:
        Returns fallback message on API failure
    """
    if not GEMINI_API_KEY:
        return _get_fallback_response(question)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = build_prompt(question, user_data, chat_history)

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 300,
                "temperature": 0.7,
            }
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

    return (
        "I'm here to help with budgeting, investments, loans, "
        "tax planning, and retirement! Try asking something "
        "specific like 'How much should I save monthly?' "
        "Note: This is general guidance. Consult a SEBI "
        "registered advisor for personalized advice."
    )


if __name__ == "__main__":
    print("=== ArthAI Chatbot Demo ===\n")

    test_user = {
        "age": 22,
        "monthly_income": 35000,
        "monthly_savings": 5000,
        "existing_loans": 0,
        "risk_appetite": "Moderate"
    }

    test_questions = [
        "Should I invest in FD or mutual funds?",
        "How much tax can I save this year?",
        "Is it better to prepay my loan or invest?"
    ]

    for q in test_questions:
        print(f"❓ User: {q}")
        response = get_ai_response(q, test_user)
        print(f"🤖 ArthAI: {response}\n")
        print("-" * 60)
