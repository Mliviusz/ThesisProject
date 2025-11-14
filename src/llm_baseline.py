
from google import genai
from google.genai import types

class GeminiChat:
      def __init__(self, constraints, model, thinking_budget):
            self.client = genai.Client(api_key="AIzaSyDA_ImQ0YihjlzqVqWvx58fHWRInqrDL_w")
            self.chat = self.client.chats.create(model=model, config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
    ))
            self.set_up_constaints(constraints)

      def set_up_constaints(self, constraints):
            full_prompt = f"{constraints}\n"
            self.send_message(full_prompt)

      def send_message(self,prompt):
            return self.chat.send_message(prompt).text.strip()
