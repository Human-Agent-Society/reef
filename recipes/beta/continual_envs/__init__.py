"""Continual environments: Harbor tasks built from the failures of live traffic, for continual training.

The policy serves traffic. Each session of the traffic is scored, with no verifier, and the failed sessions of each
round go to a generation service. The service mines environments from them, builds and reviews tasks, and keeps what
its reviews learned about building tasks. The built-in generated-task recipe trains the policy on the tasks, and the
next round reads the traffic of the trained policy.

- ``sessions``: the traffic sessions of a round's records, and the record folders that a generation run reads.
- ``generation_service``: the generation service, Reef Platform's generation API.
- ``generator``: ``TrafficTaskGenerator``: the sessions of each round to one generation run, and its tasks back.
"""

from recipes.beta.continual_envs.generator import TrafficGeneratorSettings, TrafficTaskGenerator

__all__ = ["TrafficGeneratorSettings", "TrafficTaskGenerator"]
