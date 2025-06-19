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
    from ..models.alert_result_conditions_result_item import (
        AlertResultConditionsResultItem,
    )
    from ..models.alert_result_metadata import AlertResultMetadata
    from ..models.alert_result_notification_type_0 import AlertResultNotificationType0


T = TypeVar("T", bound="AlertResult")


@_attrs_define
class AlertResult:
    """
    Attributes:
        rule_name (str):
        status (bool):
        rule_id (Union[None, Unset, str]):
        conditions_result (Union[Unset, list['AlertResultConditionsResultItem']]):
        metadata (Union[Unset, AlertResultMetadata]):
        project_name (Union[None, Unset, str]):
        project_id (Union[None, Unset, str]):
        trace_span_id (Union[None, Unset, str]):
        combine_type (Union[None, Unset, str]):
        notification (Union['AlertResultNotificationType0', None, Unset]):
    """

    rule_name: str
    status: bool
    rule_id: Union[None, Unset, str] = UNSET
    conditions_result: Union[Unset, list["AlertResultConditionsResultItem"]] = UNSET
    metadata: Union[Unset, "AlertResultMetadata"] = UNSET
    project_name: Union[None, Unset, str] = UNSET
    project_id: Union[None, Unset, str] = UNSET
    trace_span_id: Union[None, Unset, str] = UNSET
    combine_type: Union[None, Unset, str] = UNSET
    notification: Union["AlertResultNotificationType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.alert_result_notification_type_0 import (
            AlertResultNotificationType0,
        )

        rule_name = self.rule_name

        status = self.status

        rule_id: Union[None, Unset, str]
        if isinstance(self.rule_id, Unset):
            rule_id = UNSET
        else:
            rule_id = self.rule_id

        conditions_result: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.conditions_result, Unset):
            conditions_result = []
            for conditions_result_item_data in self.conditions_result:
                conditions_result_item = conditions_result_item_data.to_dict()
                conditions_result.append(conditions_result_item)

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        project_name: Union[None, Unset, str]
        if isinstance(self.project_name, Unset):
            project_name = UNSET
        else:
            project_name = self.project_name

        project_id: Union[None, Unset, str]
        if isinstance(self.project_id, Unset):
            project_id = UNSET
        else:
            project_id = self.project_id

        trace_span_id: Union[None, Unset, str]
        if isinstance(self.trace_span_id, Unset):
            trace_span_id = UNSET
        else:
            trace_span_id = self.trace_span_id

        combine_type: Union[None, Unset, str]
        if isinstance(self.combine_type, Unset):
            combine_type = UNSET
        else:
            combine_type = self.combine_type

        notification: Union[None, Unset, dict[str, Any]]
        if isinstance(self.notification, Unset):
            notification = UNSET
        elif isinstance(self.notification, AlertResultNotificationType0):
            notification = self.notification.to_dict()
        else:
            notification = self.notification

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "rule_name": rule_name,
                "status": status,
            }
        )
        if rule_id is not UNSET:
            field_dict["rule_id"] = rule_id
        if conditions_result is not UNSET:
            field_dict["conditions_result"] = conditions_result
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if project_name is not UNSET:
            field_dict["project_name"] = project_name
        if project_id is not UNSET:
            field_dict["project_id"] = project_id
        if trace_span_id is not UNSET:
            field_dict["trace_span_id"] = trace_span_id
        if combine_type is not UNSET:
            field_dict["combine_type"] = combine_type
        if notification is not UNSET:
            field_dict["notification"] = notification

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.alert_result_conditions_result_item import (
            AlertResultConditionsResultItem,
        )
        from ..models.alert_result_metadata import AlertResultMetadata
        from ..models.alert_result_notification_type_0 import (
            AlertResultNotificationType0,
        )

        d = dict(src_dict)
        rule_name = d.pop("rule_name")

        status = d.pop("status")

        def _parse_rule_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        rule_id = _parse_rule_id(d.pop("rule_id", UNSET))

        conditions_result = []
        _conditions_result = d.pop("conditions_result", UNSET)
        for conditions_result_item_data in _conditions_result or []:
            conditions_result_item = AlertResultConditionsResultItem.from_dict(
                conditions_result_item_data
            )

            conditions_result.append(conditions_result_item)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, AlertResultMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = AlertResultMetadata.from_dict(_metadata)

        def _parse_project_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        project_name = _parse_project_name(d.pop("project_name", UNSET))

        def _parse_project_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        project_id = _parse_project_id(d.pop("project_id", UNSET))

        def _parse_trace_span_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        trace_span_id = _parse_trace_span_id(d.pop("trace_span_id", UNSET))

        def _parse_combine_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        combine_type = _parse_combine_type(d.pop("combine_type", UNSET))

        def _parse_notification(
            data: object,
        ) -> Union["AlertResultNotificationType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_0 = AlertResultNotificationType0.from_dict(data)

                return notification_type_0
            except:  # noqa: E722
                pass
            return cast(Union["AlertResultNotificationType0", None, Unset], data)

        notification = _parse_notification(d.pop("notification", UNSET))

        alert_result = cls(
            rule_name=rule_name,
            status=status,
            rule_id=rule_id,
            conditions_result=conditions_result,
            metadata=metadata,
            project_name=project_name,
            project_id=project_id,
            trace_span_id=trace_span_id,
            combine_type=combine_type,
            notification=notification,
        )

        alert_result.additional_properties = d
        return alert_result

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
