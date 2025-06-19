from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.example import Example


T = TypeVar("T", bound="DatasetInsertExamples")


@_attrs_define
class DatasetInsertExamples:
    """
    Attributes:
        dataset_alias (str):
        examples (list['Example']):
        project_name (str):
    """

    dataset_alias: str
    examples: list["Example"]
    project_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_alias = self.dataset_alias

        examples = []
        for examples_item_data in self.examples:
            examples_item = examples_item_data.to_dict()
            examples.append(examples_item)

        project_name = self.project_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_alias": dataset_alias,
                "examples": examples,
                "project_name": project_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.example import Example

        d = dict(src_dict)
        dataset_alias = d.pop("dataset_alias")

        examples = []
        _examples = d.pop("examples")
        for examples_item_data in _examples:
            examples_item = Example.from_dict(examples_item_data)

            examples.append(examples_item)

        project_name = d.pop("project_name")

        dataset_insert_examples = cls(
            dataset_alias=dataset_alias,
            examples=examples,
            project_name=project_name,
        )

        dataset_insert_examples.additional_properties = d
        return dataset_insert_examples

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
