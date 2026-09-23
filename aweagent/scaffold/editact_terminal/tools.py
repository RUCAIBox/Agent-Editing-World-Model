"""Keep the evaluated Terminal tool schema without changing the runtime API."""

from textwrap import dedent

from aweagent.core.tool.protocol import Tool


class TerminalBashTool(Tool):
    def __init__(self, tool):
        self.tool = tool

    @property
    def name(self):
        return self.tool.name

    @property
    def description(self):
        # Reference prompt text. Actual shell persistence is runtime-dependent.
        return dedent("""\
            Execute a bash command in the terminal.
            * One command at a time: You can only execute one bash command at a time. \
If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
            * Persistent session: Commands execute in a persistent shell session where \
environment variables, virtual environments, and working directory persist between commands.
            * Soft timeout: Commands have a soft timeout. Once that's reached, the command \
will be interrupted.
            * Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in \
shell scripts or commands in this environment. The runtime may not support them and can \
cause unusable shell sessions. If you want to run multi-line bash commands, write the \
commands to a file and then run it, instead.
            * For commands that may run indefinitely, run them in the background and \
redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
            * Directory verification: Before creating new directories or files, first \
verify the parent directory exists and is the correct location.
            * Directory management: Try to maintain working directory by using absolute \
paths and avoiding excessive use of `cd`.
            * Output truncation: If the output exceeds a maximum length, it will be \
truncated before being returned.""")

    @property
    def parameters(self):
        parameters = self.tool.parameters
        parameters["properties"]["command"]["description"] = (
            "The bash command to execute. "
            "Can be empty string to view additional logs when previous "
            "exit code is `-1`. "
            "Can be `C-c` (Ctrl+C) to interrupt the currently running process."
        )
        return parameters

    async def execute(self, params, session=None):
        return await self.tool.execute(params, session=session)
