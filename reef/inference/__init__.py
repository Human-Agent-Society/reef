"""Concrete inference backends implementing Reef runtime contracts.

Each integration owns native engine launch, request adaptation, loading saved
weight files and its worker lifecycle. The serving runtimes of
``reef.runtime.serving`` decide which published release the engines load, and
integrations report their results through those contracts.

Import this package without loading optional inference or training frameworks.
Concrete integrations stay in their own subpackages and never import a training
backend. Service assembly selects implementations and supplies runtime resources.

``provider_protocol`` holds the provider wire format that the proxy, the model
clients and the recorders share: headers, Server-Sent Events, stream folding
and token usage. It imports only the standard library.
"""
