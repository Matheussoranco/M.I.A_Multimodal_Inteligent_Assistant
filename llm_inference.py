from openai import OpenAI
from planning.action_planner import ActionPlanner
from safety.sanitizer import ActionSanitizer

class LLMInference:
    def __init__(self, model_id='mistral:instruct', url='http://localhost:11434/v1', api_key='ollama'):
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.planner = ActionPlanner(self)
        self.sanitizer = ActionSanitizer()

    def query_model(self, text, context=None):
        """Enhanced with cognitive context and action planning"""
        system_msg = """You are M.I.A - Multimodal Intelligent Assistant. 
        Context: {context}
        Available Tools: {tools}""".format(
            context=context or "No context available",
            tools=self.planner.get_tool_descriptions()
        )

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text}
            ]
        )
        
        raw_response = response.choices[0].message.content
        return self._process_action(raw_response)

    def _process_action(self, response):
        """Handle action-oriented responses"""
        if "ACTION:" in response:
            action = response.split("ACTION:")[1].strip()
            self.sanitizer.validate_action(action)
            return f"ACTION: {action}"
        return response