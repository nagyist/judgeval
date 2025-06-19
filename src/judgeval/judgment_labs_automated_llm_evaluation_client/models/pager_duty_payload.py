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

from ..models.pager_duty_payload_severity import PagerDutyPayloadSeverity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.pager_duty_payload_custom_details_type_0 import (
        PagerDutyPayloadCustomDetailsType0,
    )


T = TypeVar("T", bound="PagerDutyPayload")


@_attrs_define
class PagerDutyPayload:
    """Payload for sending PagerDuty alerts.

    Attributes:
        summary: Brief description of the alert
        source: Source of the alert (defaults to "judgeval")
        severity: Severity level of the alert
        routing_key: PagerDuty integration routing key
        component: Optional component that triggered the alert
        group: Optional logical grouping for the alert
        class_type: Optional class/type of alert event
        custom_details: Optional dictionary of additional alert details

        Attributes:
            summary (str):
            routing_key (str):
            severity (Union[Unset, PagerDutyPayloadSeverity]):  Default: PagerDutyPayloadSeverity.ERROR.
            source (Union[Unset, str]):  Default: 'judgeval'.
            component (Union[None, Unset, str]):
            group (Union[None, Unset, str]):
            class_type (Union[None, Unset, str]):
            custom_details (Union['PagerDutyPayloadCustomDetailsType0', None, Unset]):
    """

    summary: str
    routing_key: str
    severity: Union[Unset, PagerDutyPayloadSeverity] = PagerDutyPayloadSeverity.ERROR
    source: Union[Unset, str] = "judgeval"
    component: Union[None, Unset, str] = UNSET
    group: Union[None, Unset, str] = UNSET
    class_type: Union[None, Unset, str] = UNSET
    custom_details: Union["PagerDutyPayloadCustomDetailsType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.pager_duty_payload_custom_details_type_0 import (
            PagerDutyPayloadCustomDetailsType0,
        )

        summary = self.summary

        routing_key = self.routing_key

        severity: Union[Unset, str] = UNSET
        if not isinstance(self.severity, Unset):
            severity = self.severity.value

        source = self.source

        component: Union[None, Unset, str]
        if isinstance(self.component, Unset):
            component = UNSET
        else:
            component = self.component

        group: Union[None, Unset, str]
        if isinstance(self.group, Unset):
            group = UNSET
        else:
            group = self.group

        class_type: Union[None, Unset, str]
        if isinstance(self.class_type, Unset):
            class_type = UNSET
        else:
            class_type = self.class_type

        custom_details: Union[None, Unset, dict[str, Any]]
        if isinstance(self.custom_details, Unset):
            custom_details = UNSET
        elif isinstance(self.custom_details, PagerDutyPayloadCustomDetailsType0):
            custom_details = self.custom_details.to_dict()
        else:
            custom_details = self.custom_details

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "summary": summary,
                "routing_key": routing_key,
            }
        )
        if severity is not UNSET:
            field_dict["severity"] = severity
        if source is not UNSET:
            field_dict["source"] = source
        if component is not UNSET:
            field_dict["component"] = component
        if group is not UNSET:
            field_dict["group"] = group
        if class_type is not UNSET:
            field_dict["class_type"] = class_type
        if custom_details is not UNSET:
            field_dict["custom_details"] = custom_details

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.pager_duty_payload_custom_details_type_0 import (
            PagerDutyPayloadCustomDetailsType0,
        )

        d = dict(src_dict)
        summary = d.pop("summary")

        routing_key = d.pop("routing_key")

        _severity = d.pop("severity", UNSET)
        severity: Union[Unset, PagerDutyPayloadSeverity]
        if isinstance(_severity, Unset):
            severity = UNSET
        else:
            severity = PagerDutyPayloadSeverity(_severity)

        source = d.pop("source", UNSET)

        def _parse_component(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        component = _parse_component(d.pop("component", UNSET))

        def _parse_group(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        group = _parse_group(d.pop("group", UNSET))

        def _parse_class_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        class_type = _parse_class_type(d.pop("class_type", UNSET))

        def _parse_custom_details(
            data: object,
        ) -> Union["PagerDutyPayloadCustomDetailsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                custom_details_type_0 = PagerDutyPayloadCustomDetailsType0.from_dict(
                    data
                )

                return custom_details_type_0
            except:  # noqa: E722
                pass
            return cast(Union["PagerDutyPayloadCustomDetailsType0", None, Unset], data)

        custom_details = _parse_custom_details(d.pop("custom_details", UNSET))

        pager_duty_payload = cls(
            summary=summary,
            routing_key=routing_key,
            severity=severity,
            source=source,
            component=component,
            group=group,
            class_type=class_type,
            custom_details=custom_details,
        )

        pager_duty_payload.additional_properties = d
        return pager_duty_payload

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
