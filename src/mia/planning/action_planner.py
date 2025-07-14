from langchain.agents import Tool
from langchain.chains import LLMChain

class ActionPlanner:
    def __init__(self, llm):
        self.tools = [
            Tool(
                name="WebSearch",
                func=self.web_search,
                description="Search the web for current information"
            ),
            Tool(
                name="CodeWriter",
                func=self.write_code,
                description="Generate and execute Python code"
            )
        ]
        self.llm_chain = LLMChain(llm=llm, prompt=self._create_prompt())
        
    def create_plan(self, objective):
        """Generate executable action sequence"""
        return self.llm_chain.run(
            objective=objective,
            tools=self.tools
        )
    
    def web_search(self, query):
        from duckduckgo_search import DDGS
        return [r['body'] for r in DDGS().text(query, max_results=3)]
    
    def write_code(self, task):
        return f"# Generated code for: {task}"

# Usage:
# planner = ActionPlanner(llm)
# plan = planner.create_plan("Find current AI trends")