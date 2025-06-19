from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scorer_kwargs_type_0 import ScorerKwargsType0


T = TypeVar("T", bound="Scorer")


@_attrs_define
class Scorer:
    """Data object to describe a Metric used for evaluating Examples

    Attributes:
        threshold (float):
        score_type (str):
        kwargs (Union['ScorerKwargsType0', None, Unset]):
    """

    threshold: float
    score_type: str
    kwargs: Union["ScorerKwargsType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.scorer_kwargs_type_0 import ScorerKwargsType0

        threshold = self.threshold

        score_type = self.score_type

        kwargs: Union[None, Unset, dict[str, Any]]
        if isinstance(self.kwargs, Unset):
            kwargs = UNSET
        elif isinstance(self.kwargs, ScorerKwargsType0):
            kwargs = self.kwargs.to_dict()
        else:
            kwargs = self.kwargs

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "threshold": threshold,
                "score_type": score_type,
            }
        )
        if kwargs is not UNSET:
            field_dict["kwargs"] = kwargs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.scorer_kwargs_type_0 import ScorerKwargsType0

        d = dict(src_dict)
        threshold = d.pop("threshold")

        score_type = d.pop("score_type")

        def _parse_kwargs(data: object) -> Union["ScorerKwargsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                kwargs_type_0 = ScorerKwargsType0.from_dict(data)

                return kwargs_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ScorerKwargsType0", None, Unset], data)

        kwargs = _parse_kwargs(d.pop("kwargs", UNSET))

        scorer = cls(
            threshold=threshold,
            score_type=score_type,
            kwargs=kwargs,
        )

        scorer.additional_properties = d
        return scorer

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
