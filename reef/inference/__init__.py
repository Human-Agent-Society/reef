"""Concrete inference backends implementing Reef runtime contracts.

Each integration owns native engine launch, request adaptation, weight reception
and its worker lifecycle. Reef runtime coordination selects operations and owns
publication ordering; integrations report their results through those contracts.

Import this package without loading optional inference or training frameworks.
Concrete integrations stay in their own subpackages and never import a training
backend. Service assembly selects implementations and supplies runtime resources.
"""
