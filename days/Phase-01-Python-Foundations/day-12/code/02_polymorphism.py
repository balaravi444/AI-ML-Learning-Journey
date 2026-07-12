# Program 2 - Polymorphism with AI Assistants
# Day 12 - OOP: Encapsulation, Polymorphism & Abstract Classes

class AIAssistant:
    def respond(self, question):
        print("Processing...")

    def get_specialty(self):
        print("General AI")


class CodeAssistant(AIAssistant):
    def respond(self, question):
        print(f"Writing code for: {question}")

    def get_specialty(self):
        print("I specialize in: Code Generation")


class DataAssistant(AIAssistant):
    def respond(self, question):
        print(f"Analyzing data for: {question}")

    def get_specialty(self):
        print("I specialize in: Data Analysis")


class ChatAssistant(AIAssistant):
    def respond(self, question):
        print(f"Chatting about: {question}")

    def get_specialty(self):
        print("I specialize in: Conversation")


# Polymorphism in action
coder = CodeAssistant()
analyst = DataAssistant()
chatter = ChatAssistant()

assistants = [coder, analyst, chatter]
for assistant in assistants:
    assistant.respond("machine learning")
    assistant.get_specialty()
    print("---")
