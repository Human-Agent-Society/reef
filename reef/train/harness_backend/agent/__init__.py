"""The deployment's coding agent, which a method runs in its job's workspace.

``settings`` reads ``evolution.agent`` (``AgentSettings``); ``run`` runs the
agent in a job (``AgentExecution``, ``run_agent``) behind the loopback
gateway of ``reef.harness.gateway``, which serves its model and its tools;
``method`` is ``AgentMethod``, a method that is a folder of prompts.
``protocol.md`` is Reef's rules for every run: the workspace, the outputs and
the tools.
"""
