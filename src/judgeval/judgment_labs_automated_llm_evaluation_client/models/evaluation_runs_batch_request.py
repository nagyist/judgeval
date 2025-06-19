from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.evaluation_runs_batch_request_evaluation_entries_item import (
        EvaluationRunsBatchRequestEvaluationEntriesItem,
    )


T = TypeVar("T", bound="EvaluationRunsBatchRequest")


@_attrs_define
class EvaluationRunsBatchRequest:
    """Request model for batched evaluation runs from background service

    Attributes:
        organization_id (str):
        evaluation_entries (list['EvaluationRunsBatchRequestEvaluationEntriesItem']):
    """

    organization_id: str
    evaluation_entries: list["EvaluationRunsBatchRequestEvaluationEntriesItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        organization_id = self.organization_id

        evaluation_entries = []
        for evaluation_entries_item_data in self.evaluation_entries:
            evaluation_entries_item = evaluation_entries_item_data.to_dict()
            evaluation_entries.append(evaluation_entries_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "organization_id": organization_id,
                "evaluation_entries": evaluation_entries,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.evaluation_runs_batch_request_evaluation_entries_item import (
            EvaluationRunsBatchRequestEvaluationEntriesItem,
        )

        d = dict(src_dict)
        organization_id = d.pop("organization_id")

        evaluation_entries = []
        _evaluation_entries = d.pop("evaluation_entries")
        for evaluation_entries_item_data in _evaluation_entries:
            evaluation_entries_item = (
                EvaluationRunsBatchRequestEvaluationEntriesItem.from_dict(
                    evaluation_entries_item_data
                )
            )

            evaluation_entries.append(evaluation_entries_item)

        evaluation_runs_batch_request = cls(
            organization_id=organization_id,
            evaluation_entries=evaluation_entries,
        )

        evaluation_runs_batch_request.additional_properties = d
        return evaluation_runs_batch_request

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
