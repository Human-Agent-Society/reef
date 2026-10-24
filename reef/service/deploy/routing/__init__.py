"""The managed Semantic Router deployment that ``reef serve`` owns when ``reef.routing_manifest`` is set.

Module responsibilities:
    settings: Validated settings of the owned router container.
    renderer: The router configuration and Compose file of one generation.
    registry: Per-user deployment ownership and its lock.
    containers / command_runner: Docker Compose commands, events and bounded CLI processes.
    reef_lifecycle_client: The private Unix-socket client of the REEF child's lifecycle.
    policy: The router's management API and the policy-update journal.
    deployment_controller: Startup, observation, recovery, policy updates and shutdown.
    cli: The ``validate``, ``status`` and ``cleanup`` commands.

Import each module directly; this package exports nothing, so importing it never loads the controller.
"""
