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

T = TypeVar("T", bound="ClusterDatasets")


@_attrs_define
class ClusterDatasets:
    """
    Attributes:
        project_name (str):
        dataset_aliases (Union[None, Unset, list[str]]):
    """

    project_name: str
    dataset_aliases: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        project_name = self.project_name

        dataset_aliases: Union[None, Unset, list[str]]
        if isinstance(self.dataset_aliases, Unset):
            dataset_aliases = UNSET
        elif isinstance(self.dataset_aliases, list):
            dataset_aliases = self.dataset_aliases

        else:
            dataset_aliases = self.dataset_aliases

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "project_name": project_name,
            }
        )
        if dataset_aliases is not UNSET:
            field_dict["dataset_aliases"] = dataset_aliases

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        project_name = d.pop("project_name")

        def _parse_dataset_aliases(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                dataset_aliases_type_0 = cast(list[str], data)

                return dataset_aliases_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        dataset_aliases = _parse_dataset_aliases(d.pop("dataset_aliases", UNSET))

        cluster_datasets = cls(
            project_name=project_name,
            dataset_aliases=dataset_aliases,
        )

        cluster_datasets.additional_properties = d
        return cluster_datasets

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
