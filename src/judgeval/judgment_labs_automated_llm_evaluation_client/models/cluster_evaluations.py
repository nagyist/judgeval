from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ClusterEvaluations")


@_attrs_define
class ClusterEvaluations:
    """
    Attributes:
        project_name (str):
        eval_names (Union[None, Unset, list[str]]):
    """

    project_name: str
    eval_names: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        project_name = self.project_name

        eval_names: Union[None, Unset, list[str]]
        if isinstance(self.eval_names, Unset):
            eval_names = UNSET
        elif isinstance(self.eval_names, list):
            eval_names = self.eval_names

        else:
            eval_names = self.eval_names

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "project_name": project_name,
            }
        )
        if eval_names is not UNSET:
            field_dict["eval_names"] = eval_names

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        project_name = d.pop("project_name")

        def _parse_eval_names(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                eval_names_type_0 = cast(list[str], data)

                return eval_names_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        eval_names = _parse_eval_names(d.pop("eval_names", UNSET))

        cluster_evaluations = cls(
            project_name=project_name,
            eval_names=eval_names,
        )

        cluster_evaluations.additional_properties = d
        return cluster_evaluations

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
