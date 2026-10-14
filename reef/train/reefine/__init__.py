"""Execute Reefine proposals within the shared Cordis training lifecycle.

``backend`` prepares candidates from the published serving composition.
``proposer`` builds mutations from requests and recorded failures.
``agent`` runs a coding agent, reads its workspace, and executes candidate trials.
``agent_gateway`` controls model access, call budgets, and workspace checks.
The TypeScript tools and JavaScript trial driver run inside those agent sessions.

The recipe owns prompts, defaults, provider presets, and configuration. It
injects those values into these execution components. Training code does not
import recipe policy. Independent candidate checks live in ``reef.evaluation``.
"""
