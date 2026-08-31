from generals.agents import ExpanderAgent


class FrontierAgent(ExpanderAgent):
    """Baseline expanding bot.

    Improve this class or replace it with your own implementation. The judge
    instantiates FrontierAgent() and calls act(observation, key) every turn.
    """

    def __init__(self, id: str = "FrontierAgent"):
        super().__init__(id=id)
