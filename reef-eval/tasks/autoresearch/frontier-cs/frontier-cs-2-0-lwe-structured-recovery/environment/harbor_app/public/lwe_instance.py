"""Stable public interface for structured-LWE instance consumers."""

from collections.abc import Iterator as _Iterator
from collections.abc import Sequence as _Sequence
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType

from lwe_challenge import matrix as _matrix
from lwe_challenge import schema as _schema
from lwe_challenge import verification as _verification


WitnessVerdict = _verification.WitnessVerdict

__all__ = ("Catalog", "Instance", "WitnessVerdict")


class Catalog:
    """A catalog whose entries use the stable public :class:`Instance` API."""

    __slots__ = ("__catalog_id", "__instances", "__instances_by_id")

    def __init__(self, catalog: object) -> None:
        if not isinstance(catalog, _schema.Catalog):
            raise TypeError("catalog must be loaded with Catalog.load")
        instances = tuple(Instance(spec) for spec in catalog.instances)
        object.__setattr__(self, "_Catalog__catalog_id", catalog.catalog_id)
        object.__setattr__(self, "_Catalog__instances", instances)
        object.__setattr__(
            self,
            "_Catalog__instances_by_id",
            _MappingProxyType(
                {instance.instance_id: instance for instance in instances}
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Catalog is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Catalog is immutable")

    @classmethod
    def load(cls, path: str | _Path) -> "Catalog":
        return cls(_schema.Catalog.load(path))

    @classmethod
    def load_fd(cls, descriptor: int, catalog_format: str) -> "Catalog":
        return cls(_schema.Catalog.load_fd(descriptor, catalog_format))

    @property
    def catalog_id(self) -> str:
        return self.__catalog_id

    @property
    def instances(self) -> tuple["Instance", ...]:
        return self.__instances

    def get(self, instance_id: str) -> "Instance":
        return self.__instances_by_id[instance_id]


class Instance:
    """Read-only public view of one challenge instance."""

    __slots__ = ("__spec",)

    def __init__(self, spec: object) -> None:
        if not isinstance(spec, _schema.InstanceSpec):
            raise TypeError("instances are created by Catalog.load")
        object.__setattr__(self, "_Instance__spec", spec)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Instance is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Instance is immutable")

    @property
    def instance_id(self) -> str:
        return self.__spec.instance_id

    @property
    def n(self) -> int:
        return self.__spec.n

    @property
    def m(self) -> int:
        return self.__spec.m

    @property
    def q(self) -> int:
        return self.__spec.q

    @property
    def b(self) -> tuple[int, ...]:
        return self.__spec.b

    @property
    def matrix_kind(self) -> str:
        return self.__spec.matrix.kind

    @property
    def matrix_seed_hex(self) -> str:
        return self.__spec.matrix.seed_hex

    @property
    def matrix_expansion_domain(self) -> str:
        return self.__spec.matrix.expansion_domain

    @property
    def matrix_alphabet(self) -> tuple[int, ...]:
        return self.__spec.matrix.alphabet

    @property
    def matrix_row_weight(self) -> int | None:
        return self.__spec.matrix.row_weight

    @property
    def secret_distribution_kind(self) -> str:
        return self.__spec.secret_distribution.kind

    @property
    def secret_predicate_kind(self) -> str:
        return self.__spec.secret.kind

    @property
    def secret_alphabet(self) -> tuple[int, ...]:
        return self.__spec.secret.alphabet

    @property
    def secret_weight(self) -> int | None:
        return self.__spec.secret_distribution.weight

    @property
    def secret_eta(self) -> int | None:
        return self.__spec.secret_distribution.eta

    @property
    def secret_min_nonzero(self) -> int:
        return self.__spec.secret.min_nonzero

    @property
    def secret_max_nonzero(self) -> int:
        return self.__spec.secret.max_nonzero

    @property
    def error_distribution_kind(self) -> str:
        return self.__spec.error_distribution.kind

    @property
    def error_sigma(self) -> float | None:
        return self.__spec.error_distribution.sigma

    @property
    def error_eta(self) -> int | None:
        return self.__spec.error_distribution.eta

    @property
    def error_bound(self) -> int:
        return self.__spec.error_distribution.bound

    @property
    def error_weight(self) -> int | None:
        return self.__spec.error_distribution.weight

    @property
    def error_max_abs(self) -> int:
        return self.__spec.error.max_abs

    @property
    def error_max_l1(self) -> int | None:
        return self.__spec.error.max_l1

    @property
    def error_max_l2_squared(self) -> int | None:
        return self.__spec.error.max_l2_squared

    @property
    def error_max_nonzero(self) -> int | None:
        return self.__spec.error.max_nonzero

    @property
    def instance_digest(self) -> str:
        return self.__spec.instance_digest

    def iter_rows(self) -> _Iterator[tuple[int, ...]]:
        return _matrix.iter_rows(self.__spec)

    def materialize_row_block(
        self, start: int, stop: int
    ) -> tuple[tuple[int, ...], ...]:
        return _matrix.materialize_row_block(self.__spec, start, stop)

    def materialize_rows(self) -> tuple[tuple[int, ...], ...]:
        return _matrix.materialize_rows(self.__spec)

    def matvec(self, secret: _Sequence[int]) -> tuple[int, ...]:
        return _matrix.matvec_mod(self.__spec, secret)

    def validate_secret(self, secret: _Sequence[int]) -> WitnessVerdict:
        return _verification.validate_secret(self.__spec, secret)
