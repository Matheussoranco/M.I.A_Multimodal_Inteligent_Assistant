from datetime import datetime
from mia.core.skills import BaseSkill, SkillManifest, tool
from mia.utils.safe_arithmetic import safe_eval_arithmetic

class BasicToolsSkill(BaseSkill):
    """
    Core set of basic tools that M.I.A should always have.
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="basic_tools",
            version="1.0.0",
            description="Essential utilities like time, date, and basic math.",
            author="M.I.A Team"
        )

    @tool(name="get_current_time", description="Returns the current server time.")
    def get_time(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tool(name="calculate", description="Performs basic arithmetic.")
    def calculate(self, expression: str) -> str:
        try:
            return str(safe_eval_arithmetic(expression))
        except Exception as e:
            return f"Error calculating: {e}"

    async def on_load(self):
        print("Basic Tools Skill Loaded! ready to help.")
