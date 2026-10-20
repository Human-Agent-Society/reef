"""The managed Semantic Router deployment that ``reef serve`` owns when ``reef.routing_manifest`` is set.

Module responsibilities:
    routing_config: Validated settings of the owned router container.
    routing_renderer: The router configuration and Compose file of one generation.
    routing_registry: Per-user deployment ownership and its lock.
    routing_containers / routing_command: Docker Compose commands, events and bounded CLI processes.
    routing_control: The private Unix-socket client of the REEF child's lifecycle.
    routing_policy: The router's management API and the policy-update journal.
    routing_controller: Startup, observation, recovery, policy updates and shutdown.
    routing_ops: The ``validate``, ``status`` and ``cleanup`` commands.

Import each module directly; this package exports nothing, so importing it never loads the controller.
"""
