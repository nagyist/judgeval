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
    from ..models.trace_rules_type_0 import TraceRulesType0
    from ..models.trace_span import TraceSpan


T = TypeVar("T", bound="Trace")


@_attrs_define
class Trace:
    """
    Attributes:
        trace_id (str):
        name (str):
        created_at (str):
        duration (float):
        trace_spans (list['TraceSpan']):
        overwrite (Union[Unset, bool]):  Default: False.
        offline_mode (Union[Unset, bool]):  Default: False.
        rules (Union['TraceRulesType0', None, Unset]):
        has_notification (Union[None, Unset, bool]):  Default: False.
        customer_id (Union[None, Unset, str]):
        tags (Union[None, Unset, list[str]]):
    """

    trace_id: str
    name: str
    created_at: str
    duration: float
    trace_spans: list["TraceSpan"]
    overwrite: Union[Unset, bool] = False
    offline_mode: Union[Unset, bool] = False
    rules: Union["TraceRulesType0", None, Unset] = UNSET
    has_notification: Union[None, Unset, bool] = False
    customer_id: Union[None, Unset, str] = UNSET
    tags: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.trace_rules_type_0 import TraceRulesType0

        trace_id = self.trace_id

        name = self.name

        created_at = self.created_at

        duration = self.duration

        trace_spans = []
        for trace_spans_item_data in self.trace_spans:
            trace_spans_item = trace_spans_item_data.to_dict()
            trace_spans.append(trace_spans_item)

        overwrite = self.overwrite

        offline_mode = self.offline_mode

        rules: Union[None, Unset, dict[str, Any]]
        if isinstance(self.rules, Unset):
            rules = UNSET
        elif isinstance(self.rules, TraceRulesType0):
            rules = self.rules.to_dict()
        else:
            rules = self.rules

        has_notification: Union[None, Unset, bool]
        if isinstance(self.has_notification, Unset):
            has_notification = UNSET
        else:
            has_notification = self.has_notification

        customer_id: Union[None, Unset, str]
        if isinstance(self.customer_id, Unset):
            customer_id = UNSET
        else:
            customer_id = self.customer_id

        tags: Union[None, Unset, list[str]]
        if isinstance(self.tags, Unset):
            tags = UNSET
        elif isinstance(self.tags, list):
            tags = self.tags

        else:
            tags = self.tags

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "trace_id": trace_id,
                "name": name,
                "created_at": created_at,
                "duration": duration,
                "trace_spans": trace_spans,
            }
        )
        if overwrite is not UNSET:
            field_dict["overwrite"] = overwrite
        if offline_mode is not UNSET:
            field_dict["offline_mode"] = offline_mode
        if rules is not UNSET:
            field_dict["rules"] = rules
        if has_notification is not UNSET:
            field_dict["has_notification"] = has_notification
        if customer_id is not UNSET:
            field_dict["customer_id"] = customer_id
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trace_rules_type_0 import TraceRulesType0
        from ..models.trace_span import TraceSpan

        d = dict(src_dict)
        trace_id = d.pop("trace_id")

        name = d.pop("name")

        created_at = d.pop("created_at")

        duration = d.pop("duration")

        trace_spans = []
        _trace_spans = d.pop("trace_spans")
        for trace_spans_item_data in _trace_spans:
            trace_spans_item = TraceSpan.from_dict(trace_spans_item_data)

            trace_spans.append(trace_spans_item)

        overwrite = d.pop("overwrite", UNSET)

        offline_mode = d.pop("offline_mode", UNSET)

        def _parse_rules(data: object) -> Union["TraceRulesType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                rules_type_0 = TraceRulesType0.from_dict(data)

                return rules_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceRulesType0", None, Unset], data)

        rules = _parse_rules(d.pop("rules", UNSET))

        def _parse_has_notification(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        has_notification = _parse_has_notification(d.pop("has_notification", UNSET))

        def _parse_customer_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        customer_id = _parse_customer_id(d.pop("customer_id", UNSET))

        def _parse_tags(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tags_type_0 = cast(list[str], data)

                return tags_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        tags = _parse_tags(d.pop("tags", UNSET))

        trace = cls(
            trace_id=trace_id,
            name=name,
            created_at=created_at,
            duration=duration,
            trace_spans=trace_spans,
            overwrite=overwrite,
            offline_mode=offline_mode,
            rules=rules,
            has_notification=has_notification,
            customer_id=customer_id,
            tags=tags,
        )

        trace.additional_properties = d
        return trace

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
