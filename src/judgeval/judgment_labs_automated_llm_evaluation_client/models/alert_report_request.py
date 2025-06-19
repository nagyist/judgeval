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

T = TypeVar("T", bound="AlertReportRequest")


@_attrs_define
class AlertReportRequest:
    """
    Attributes:
        start_date (str):
        end_date (Union[None, Unset, str]):
        project_name (Union[None, Unset, str]):
        comparison_period (Union[None, Unset, bool]):  Default: False.
    """

    start_date: str
    end_date: Union[None, Unset, str] = UNSET
    project_name: Union[None, Unset, str] = UNSET
    comparison_period: Union[None, Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start_date = self.start_date

        end_date: Union[None, Unset, str]
        if isinstance(self.end_date, Unset):
            end_date = UNSET
        else:
            end_date = self.end_date

        project_name: Union[None, Unset, str]
        if isinstance(self.project_name, Unset):
            project_name = UNSET
        else:
            project_name = self.project_name

        comparison_period: Union[None, Unset, bool]
        if isinstance(self.comparison_period, Unset):
            comparison_period = UNSET
        else:
            comparison_period = self.comparison_period

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "start_date": start_date,
            }
        )
        if end_date is not UNSET:
            field_dict["end_date"] = end_date
        if project_name is not UNSET:
            field_dict["project_name"] = project_name
        if comparison_period is not UNSET:
            field_dict["comparison_period"] = comparison_period

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        start_date = d.pop("start_date")

        def _parse_end_date(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        end_date = _parse_end_date(d.pop("end_date", UNSET))

        def _parse_project_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        project_name = _parse_project_name(d.pop("project_name", UNSET))

        def _parse_comparison_period(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        comparison_period = _parse_comparison_period(d.pop("comparison_period", UNSET))

        alert_report_request = cls(
            start_date=start_date,
            end_date=end_date,
            project_name=project_name,
            comparison_period=comparison_period,
        )

        alert_report_request.additional_properties = d
        return alert_report_request

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
