"""The Semantic Router integration: a scenario's requests go to external models that a router selects.

``manifest`` reads and checks the routing manifest of a deployment, and validates each request against it before
scenario resolution. ``preview`` asks the router which target an automatic request selects. The other modules
manage the Semantic Router of a managed deployment:

    settings: Validated settings of the router, and the settings of the REEF service that its controller reads.
    renderer: The router configuration and Compose file of one generation.
    registry: Per-user deployment ownership and its lock.
    containers / command_runner: Docker Compose commands, events and bounded CLI processes.
    reef_lifecycle_client: The private Unix-socket client of the REEF child's lifecycle.
    policy: The router's management API and the policy-update journal.
    deployment_controller: Startup, observation, recovery, policy updates and shutdown.

Nothing in this package imports ``reef.api`` or ``reef.deploy``. They import this package: the routed chat route,
the routed inference and ``reef serve`` use these modules, and ``reef.deploy.routing.cli`` holds the commands.

Import each module directly; this package exports nothing, so importing it never loads the controller.
"""
