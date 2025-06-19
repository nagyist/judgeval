from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DatasetDeleteExamples")


@_attrs_define
class DatasetDeleteExamples:
    """
    Attributes:
        dataset_alias (str):
        example_ids (list[str]):
    """

    dataset_alias: str
    example_ids: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_alias = self.dataset_alias

        example_ids = self.example_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_alias": dataset_alias,
                "example_ids": example_ids,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        dataset_alias = d.pop("dataset_alias")

        example_ids = cast(list[str], d.pop("example_ids"))

        dataset_delete_examples = cls(
            dataset_alias=dataset_alias,
            example_ids=example_ids,
        )

        dataset_delete_examples.additional_properties = d
        return dataset_delete_examples

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
