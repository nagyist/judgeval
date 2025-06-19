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

T = TypeVar("T", bound="EmailReportRequest")


@_attrs_define
class EmailReportRequest:
    """Request model for generating and sending an email report

    Attributes:
        start_date (str):
        email_addresses (list[str]):
        end_date (Union[None, Unset, str]):
        project_name (Union[None, Unset, str]):
        comparison_period (Union[None, Unset, bool]):  Default: False.
        subject (Union[None, Unset, str]):
    """

    start_date: str
    email_addresses: list[str]
    end_date: Union[None, Unset, str] = UNSET
    project_name: Union[None, Unset, str] = UNSET
    comparison_period: Union[None, Unset, bool] = False
    subject: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start_date = self.start_date

        email_addresses = self.email_addresses

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

        subject: Union[None, Unset, str]
        if isinstance(self.subject, Unset):
            subject = UNSET
        else:
            subject = self.subject

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "start_date": start_date,
                "email_addresses": email_addresses,
            }
        )
        if end_date is not UNSET:
            field_dict["end_date"] = end_date
        if project_name is not UNSET:
            field_dict["project_name"] = project_name
        if comparison_period is not UNSET:
            field_dict["comparison_period"] = comparison_period
        if subject is not UNSET:
            field_dict["subject"] = subject

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        start_date = d.pop("start_date")

        email_addresses = cast(list[str], d.pop("email_addresses"))

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

        def _parse_subject(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        subject = _parse_subject(d.pop("subject", UNSET))

        email_report_request = cls(
            start_date=start_date,
            email_addresses=email_addresses,
            end_date=end_date,
            project_name=project_name,
            comparison_period=comparison_period,
            subject=subject,
        )

        email_report_request.additional_properties = d
        return email_report_request

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
