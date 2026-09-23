"""SearchSWE actor with Doc2Repo AEWM judgment and revision."""

from aweagent.scaffold.editact.agent import EditActAgent
from aweagent.scaffold.search_swe.agent import SearchSWEAgent


class EditActSWEAgent(EditActAgent):
    domain = "swe"
    actor_class = SearchSWEAgent
