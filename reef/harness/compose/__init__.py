"""compose: the cordis calculus core as Python.

A Context tree composes plugins whose every side effect is tracked with its
inverse, so disposal replays inverses LIFO and the tree can be torn down or
partially retired without restarts. Services pass between plugins through
declared inject/provide capabilities, enforced at the point of use and kept
consistent reactively: a fiber whose provider is missing waits, one whose
provider appears loads, and one whose provider changes or withdraws unloads
and reloads. ``.loader`` reconciles a declarative config tree
onto live fibers. See UPSTREAM.md for the conformance map to
the reference implementation and the paper.

The package assumes one thread and, when asyncio is used, one event loop;
free-threaded Python and multi-loop use are out of scope.
"""

from .context import Context, RealmKey, ServiceAccessError
from .events import EventName
from .fiber import EffectMeta, FiberState, InactiveEffectError, ValidationError
from .registry import CycleError, MissingProviderError
from .service import Service

__all__ = [
    "Context",
    "CycleError",
    "EffectMeta",
    "EventName",
    "FiberState",
    "InactiveEffectError",
    "MissingProviderError",
    "RealmKey",
    "Service",
    "ServiceAccessError",
    "ValidationError",
]
