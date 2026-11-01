"""ReasoningBank (Ouyang et al., arXiv 2509.25140) as a Reef harness evolution method.

A deployment names ``recipes.reasoning_bank:ReasoningBank`` in ``evolution.method``. ``method`` holds the method,
``prompts/`` the judge and extractor prompts, and ``examples/serve.yaml`` an online deployment on pi.
"""

from recipes.reasoning_bank.method import ReasoningBank

__all__ = ["ReasoningBank"]
