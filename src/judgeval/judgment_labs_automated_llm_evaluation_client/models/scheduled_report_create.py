from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.schedule_frequency import ScheduleFrequency
from ..types import UNSET, Unset

T = TypeVar("T", bound="ScheduledReportCreate")


@_attrs_define
class ScheduledReportCreate:
    """
    Attributes:
        name (str):
        email_addresses (list[str]):
        frequency (Union[ScheduleFrequency, str]):
        hour (int):
        minute (int):
        project_name (Union[None, Unset, str]):
        subject (Union[None, Unset, str]):
        day_of_week (Union[None, Unset, list[int]]):
        day_of_month (Union[None, Unset, int]):
        timezone (Union[None, Unset, str]):  Default: 'UTC'.
        comparison_period (Union[Unset, bool]):  Default: False.
        time_range_days (Union[Unset, int]):  Default: 7.
        active (Union[Unset, bool]):  Default: True.
    """

    name: str
    email_addresses: list[str]
    frequency: Union[ScheduleFrequency, str]
    hour: int
    minute: int
    project_name: Union[None, Unset, str] = UNSET
    subject: Union[None, Unset, str] = UNSET
    day_of_week: Union[None, Unset, list[int]] = UNSET
    day_of_month: Union[None, Unset, int] = UNSET
    timezone: Union[None, Unset, str] = "UTC"
    comparison_period: Union[Unset, bool] = False
    time_range_days: Union[Unset, int] = 7
    active: Union[Unset, bool] = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        email_addresses = self.email_addresses

        frequency: str
        if isinstance(self.frequency, ScheduleFrequency):
            frequency = self.frequency.value
        else:
            frequency = self.frequency

        hour = self.hour

        minute = self.minute

        project_name: Union[None, Unset, str]
        if isinstance(self.project_name, Unset):
            project_name = UNSET
        else:
            project_name = self.project_name

        subject: Union[None, Unset, str]
        if isinstance(self.subject, Unset):
            subject = UNSET
        else:
            subject = self.subject

        day_of_week: Union[None, Unset, list[int]]
        if isinstance(self.day_of_week, Unset):
            day_of_week = UNSET
        elif isinstance(self.day_of_week, list):
            day_of_week = self.day_of_week

        else:
            day_of_week = self.day_of_week

        day_of_month: Union[None, Unset, int]
        if isinstance(self.day_of_month, Unset):
            day_of_month = UNSET
        else:
            day_of_month = self.day_of_month

        timezone: Union[None, Unset, str]
        if isinstance(self.timezone, Unset):
            timezone = UNSET
        else:
            timezone = self.timezone

        comparison_period = self.comparison_period

        time_range_days = self.time_range_days

        active = self.active

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "email_addresses": email_addresses,
                "frequency": frequency,
                "hour": hour,
                "minute": minute,
            }
        )
        if project_name is not UNSET:
            field_dict["project_name"] = project_name
        if subject is not UNSET:
            field_dict["subject"] = subject
        if day_of_week is not UNSET:
            field_dict["day_of_week"] = day_of_week
        if day_of_month is not UNSET:
            field_dict["day_of_month"] = day_of_month
        if timezone is not UNSET:
            field_dict["timezone"] = timezone
        if comparison_period is not UNSET:
            field_dict["comparison_period"] = comparison_period
        if time_range_days is not UNSET:
            field_dict["time_range_days"] = time_range_days
        if active is not UNSET:
            field_dict["active"] = active

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        email_addresses = cast(list[str], d.pop("email_addresses"))

        def _parse_frequency(data: object) -> Union[ScheduleFrequency, str]:
            try:
                if not isinstance(data, str):
                    raise TypeError()
                frequency_type_1 = ScheduleFrequency(data)

                return frequency_type_1
            except:  # noqa: E722
                pass
            return cast(Union[ScheduleFrequency, str], data)

        frequency = _parse_frequency(d.pop("frequency"))

        hour = d.pop("hour")

        minute = d.pop("minute")

        def _parse_project_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        project_name = _parse_project_name(d.pop("project_name", UNSET))

        def _parse_subject(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        subject = _parse_subject(d.pop("subject", UNSET))

        def _parse_day_of_week(data: object) -> Union[None, Unset, list[int]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                day_of_week_type_0 = cast(list[int], data)

                return day_of_week_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[int]], data)

        day_of_week = _parse_day_of_week(d.pop("day_of_week", UNSET))

        def _parse_day_of_month(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        day_of_month = _parse_day_of_month(d.pop("day_of_month", UNSET))

        def _parse_timezone(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        timezone = _parse_timezone(d.pop("timezone", UNSET))

        comparison_period = d.pop("comparison_period", UNSET)

        time_range_days = d.pop("time_range_days", UNSET)

        active = d.pop("active", UNSET)

        scheduled_report_create = cls(
            name=name,
            email_addresses=email_addresses,
            frequency=frequency,
            hour=hour,
            minute=minute,
            project_name=project_name,
            subject=subject,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            timezone=timezone,
            comparison_period=comparison_period,
            time_range_days=time_range_days,
            active=active,
        )

        scheduled_report_create.additional_properties = d
        return scheduled_report_create

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
