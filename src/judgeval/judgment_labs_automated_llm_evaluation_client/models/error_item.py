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

T = TypeVar("T", bound="ErrorItem")


@_attrs_define
class ErrorItem:
    """
    Attributes:
        id (str):
        project_id (str):
        project_name (str):
        trace_id (str):
        span_id (str):
        error_type (str):
        error_message (str):
        error_traceback (str):
        created_at (str):
        timestamp (str):
        trace_name (Union[None, Unset, str]):
        span_type (Union[None, Unset, str]):
        function_name (Union[None, Unset, str]):
    """

    id: str
    project_id: str
    project_name: str
    trace_id: str
    span_id: str
    error_type: str
    error_message: str
    error_traceback: str
    created_at: str
    timestamp: str
    trace_name: Union[None, Unset, str] = UNSET
    span_type: Union[None, Unset, str] = UNSET
    function_name: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        project_id = self.project_id

        project_name = self.project_name

        trace_id = self.trace_id

        span_id = self.span_id

        error_type = self.error_type

        error_message = self.error_message

        error_traceback = self.error_traceback

        created_at = self.created_at

        timestamp = self.timestamp

        trace_name: Union[None, Unset, str]
        if isinstance(self.trace_name, Unset):
            trace_name = UNSET
        else:
            trace_name = self.trace_name

        span_type: Union[None, Unset, str]
        if isinstance(self.span_type, Unset):
            span_type = UNSET
        else:
            span_type = self.span_type

        function_name: Union[None, Unset, str]
        if isinstance(self.function_name, Unset):
            function_name = UNSET
        else:
            function_name = self.function_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "project_id": project_id,
                "project_name": project_name,
                "trace_id": trace_id,
                "span_id": span_id,
                "error_type": error_type,
                "error_message": error_message,
                "error_traceback": error_traceback,
                "created_at": created_at,
                "timestamp": timestamp,
            }
        )
        if trace_name is not UNSET:
            field_dict["trace_name"] = trace_name
        if span_type is not UNSET:
            field_dict["span_type"] = span_type
        if function_name is not UNSET:
            field_dict["function_name"] = function_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        project_id = d.pop("project_id")

        project_name = d.pop("project_name")

        trace_id = d.pop("trace_id")

        span_id = d.pop("span_id")

        error_type = d.pop("error_type")

        error_message = d.pop("error_message")

        error_traceback = d.pop("error_traceback")

        created_at = d.pop("created_at")

        timestamp = d.pop("timestamp")

        def _parse_trace_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        trace_name = _parse_trace_name(d.pop("trace_name", UNSET))

        def _parse_span_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        span_type = _parse_span_type(d.pop("span_type", UNSET))

        def _parse_function_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        function_name = _parse_function_name(d.pop("function_name", UNSET))

        error_item = cls(
            id=id,
            project_id=project_id,
            project_name=project_name,
            trace_id=trace_id,
            span_id=span_id,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            created_at=created_at,
            timestamp=timestamp,
            trace_name=trace_name,
            span_type=span_type,
            function_name=function_name,
        )

        error_item.additional_properties = d
        return error_item

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
