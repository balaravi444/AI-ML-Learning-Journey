"""
Day 86 — AI Agents & Tool Use
Topic: ReAct Agent from Scratch
Date: 12 August 2026
Author: Bala Ravi

Building an AI Agent that reasons and acts.
Foundation for MemoryOS agent layer!
"""
import os
import math
import warnings
warnings.filterwarnings('ignore')


# ─── Tool Definitions ─────────────────────────────
class Tool:
    """Base class for agent tools."""

    def __init__(self,
                  name: str,
                  description: str) -> None:
        self.name = name
        self.description = description

    def run(self, input_str: str) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    """Safe math calculator tool."""

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "Evaluate mathematical expressions. "
                "Input: math expression as string. "
                "Example: '2 + 2', '15 * 7', 'sqrt(144)'"))

    def run(self, input_str: str) -> str:
        try:
            # Safe eval with math functions only
            allowed = {
                'abs': abs, 'round': round,
                'min': min, 'max': max,
                'sqrt': math.sqrt,
                'pow': math.pow,
                'log': math.log,
                'pi': math.pi, 'e': math.e}
            result = eval(
                input_str.strip(),
                {"__builtins__": {}},
                allowed)
            return str(round(float(result), 6))
        except Exception as e:
            return f"Error: {str(e)}"


class KnowledgeBaseTool(Tool):
    """Search a local knowledge base."""

    def __init__(self,
                  knowledge: dict) -> None:
        super().__init__(
            name="knowledge_base",
            description=(
                "Search the knowledge base for "
                "information. Input: search query. "
                "Returns relevant information."))
        self.knowledge = knowledge

    def run(self, query: str) -> str:
        query_lower = query.lower()
        results = []

        for key, value in self.knowledge.items():
            if any(
                    word in key.lower() or
                    word in value.lower()
                    for word in query_lower.split()):
                results.append(f"{key}: {value}")

        if results:
            return "\n".join(results[:3])
        return "No relevant information found."


class WordCountTool(Tool):
    """Count words in text."""

    def __init__(self) -> None:
        super().__init__(
            name="word_count",
            description=(
                "Count words, characters, or sentences "
                "in text. Input: text to analyze."))

    def run(self, input_str: str) -> str:
        words = len(input_str.split())
        chars = len(input_str)
        sentences = input_str.count('.') + \
            input_str.count('!') + \
            input_str.count('?')
        return (f"Words: {words}, "
                f"Characters: {chars}, "
                f"Sentences: {sentences}")


# ─── ReAct Agent ──────────────────────────────────
class ReActAgent:
    """
    ReAct Agent — Reason + Act loop.

    Thought → Action → Observation → repeat
    until Final Answer or max iterations.
    """

    def __init__(self,
                  tools: list,
                  max_iterations: int = 5) -> None:
        """
        Initialize ReAct Agent.

        Args:
            tools: List of Tool objects
            max_iterations: Max steps before stopping
        """
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        """Build agent system prompt."""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()])

        return f"""You are a helpful AI assistant that
solves problems step by step using available tools.

Available Tools:
{tool_descriptions}

Format your response as:
Thought: [your reasoning about what to do next]
Action: tool_name
Action Input: [input for the tool]

When you have the final answer:
Thought: I now have enough information
Final Answer: [your complete answer]

Always think before acting.
Use tools when you need external information."""

    def _parse_agent_output(
            self,
            text: str) -> dict:
        """Parse agent's Thought/Action/Answer."""
        result = {
            'thought': '',
            'action': None,
            'action_input': None,
            'final_answer': None
        }

        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('Thought:'):
                result['thought'] = line[8:].strip()
            elif line.startswith('Action:'):
                result['action'] = (
                    line[7:].strip().lower())
            elif line.startswith('Action Input:'):
                result['action_input'] = (
                    line[13:].strip())
            elif line.startswith('Final Answer:'):
                result['final_answer'] = (
                    line[13:].strip())

        return result

    def _call_llm(self,
                   messages: list) -> str:
        """
        Call LLM with message history.
        Uses Gemini if available, else simulates.
        """
        try:
            import google.generativeai as genai
            api_key = os.environ.get(
                'GEMINI_API_KEY', '')
            if not api_key or api_key == 'your-api-key-here':
                raise ValueError("No API key")

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                'gemini-1.5-flash')

            # Convert messages to Gemini format
            conversation = []
            for msg in messages:
                conversation.append(msg['content'])

            prompt = "\n\n".join(conversation)
            response = model.generate_content(prompt)
            return response.text

        except Exception:
            # Simulate agent reasoning for demo
            last_msg = messages[-1]['content']
            if 'calculate' in last_msg.lower() or \
                    'math' in last_msg.lower() or \
                    'what is' in last_msg.lower() and \
                    any(c.isdigit()
                        for c in last_msg):
                return ("Thought: I need to calculate "
                        "this mathematically\n"
                        "Action: calculator\n"
                        "Action Input: 2 + 2")
            elif 'search' in last_msg.lower() or \
                    'find' in last_msg.lower():
                return ("Thought: I need to search "
                        "the knowledge base\n"
                        "Action: knowledge_base\n"
                        "Action Input: " +
                        last_msg.split()[-1])
            else:
                return ("Thought: I can answer "
                        "this directly\n"
                        "Final Answer: "
                        "Based on my knowledge, "
                        "the answer is 42.")

    def run(self, task: str) -> dict:
        """
        Run agent on a task.

        ReAct loop:
        1. Think about what to do
        2. Act (use a tool)
        3. Observe result
        4. Repeat until Final Answer

        Args:
            task: User's task/question

        Returns:
            Dictionary with answer + trace
        """
        print(f"\n{'='*50}")
        print(f"Agent Task: {task}")
        print(f"{'='*50}")

        messages = [
            {
                'role': 'system',
                'content': self._build_system_prompt()
            },
            {
                'role': 'user',
                'content': f"Task: {task}"
            }
        ]

        trace = []
        final_answer = None

        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}]")

            # Get agent response
            response = self._call_llm(messages)
            parsed = self._parse_agent_output(response)

            print(f"Thought: {parsed['thought']}")

            trace.append({
                'iteration': iteration + 1,
                'thought': parsed['thought'],
                'action': parsed['action'],
                'action_input': parsed['action_input'],
                'observation': None,
                'final_answer': parsed['final_answer']
            })

            # Check for final answer
            if parsed['final_answer']:
                final_answer = parsed['final_answer']
                print(f"Final Answer: {final_answer}")
                break

            # Execute tool
            if parsed['action']:
                tool_name = parsed['action']
                tool_input = parsed[
                    'action_input'] or ""

                print(f"Action: {tool_name}")
                print(f"Input:  {tool_input}")

                if tool_name in self.tools:
                    observation = self.tools[
                        tool_name].run(tool_input)
                else:
                    observation = (
                        f"Tool '{tool_name}' not found. "
                        f"Available: "
                        f"{list(self.tools.keys())}")

                print(f"Observation: {observation}")
                trace[-1]['observation'] = observation

                # Add observation to messages
                messages.append({
                    'role': 'assistant',
                    'content': response
                })
                messages.append({
                    'role': 'user',
                    'content': (
                        f"Observation: {observation}\n"
                        f"Continue solving the task.")
                })
            else:
                # No action specified
                final_answer = parsed.get(
                    'thought', 'Unable to complete task')
                break

        if not final_answer:
            final_answer = (
                "Max iterations reached. "
                "Partial information gathered.")

        return {
            'task': task,
            'final_answer': final_answer,
            'iterations': len(trace),
            'trace': trace
        }


def demonstrate_agents() -> None:
    """Show AI agent solving tasks."""
    print("=== AI Agent Demo ===\n")

    # Knowledge base
    knowledge = {
        'RAG': (
            'Retrieval Augmented Generation combines '
            'retrieval with LLM generation'),
        'MemoryOS': (
            'AI-powered personal knowledge OS '
            'that acts as long-term memory'),
        'ChromaDB': (
            'Open-source vector database for '
            'semantic similarity search'),
        'MobileNetV2': (
            'Lightweight CNN with 3.4M parameters '
            'optimized for mobile deployment'),
        'Transformer': (
            'Neural network using self-attention '
            'mechanism from Attention Is All You Need')
    }

    tools = [
        CalculatorTool(),
        KnowledgeBaseTool(knowledge),
        WordCountTool()
    ]

    agent = ReActAgent(tools=tools,
                        max_iterations=4)

    tasks = [
        "What is RAG and how does it relate to MemoryOS?",
        "Calculate: what is 15 multiplied by 87 plus 42?",
        "Count the words in: 'Building AI products from scratch in 90 days'"
    ]

    for task in tasks:
        result = agent.run(task)
        print(f"\n✅ Answer: {result['final_answer']}")
        print(f"   Steps taken: {result['iterations']}")


if __name__ == "__main__":
    demonstrate_agents()
