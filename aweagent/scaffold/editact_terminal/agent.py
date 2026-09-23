"""CalibForge actor with pre-execution AEWM judgment and revision."""

from aweagent.scaffold.calibforge.agent import CalibForgeAgent
from aweagent.scaffold.editact.agent import EditActAgent
from aweagent.scaffold.editact_terminal.tools import TerminalBashTool


class EditActTerminalAgent(EditActAgent):
    domain = "terminal"
    actor_class = CalibForgeAgent

    def __init__(self, actor, settings):
        super().__init__(actor, settings)
        actor._tools = [
            TerminalBashTool(tool) if tool.name == "execute_bash" else tool
            for tool in actor.get_tools()
        ]
