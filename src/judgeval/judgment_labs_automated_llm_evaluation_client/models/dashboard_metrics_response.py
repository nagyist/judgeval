from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.dashboard_metrics_response_llmusage import (
        DashboardMetricsResponseLlmusage,
    )
    from ..models.dashboard_metrics_response_project_usage_item import (
        DashboardMetricsResponseProjectUsageItem,
    )
    from ..models.dashboard_metrics_response_summary import (
        DashboardMetricsResponseSummary,
    )
    from ..models.dashboard_metrics_response_tokenbreakdown import (
        DashboardMetricsResponseTokenbreakdown,
    )
    from ..models.dashboard_metrics_response_tool_usage_item import (
        DashboardMetricsResponseToolUsageItem,
    )
    from ..models.dashboard_metrics_response_user_usage_item import (
        DashboardMetricsResponseUserUsageItem,
    )


T = TypeVar("T", bound="DashboardMetricsResponse")


@_attrs_define
class DashboardMetricsResponse:
    """
    Attributes:
        summary (DashboardMetricsResponseSummary):
        trace_count (int):
        llm_usage (DashboardMetricsResponseLlmusage):
        project_usage (list['DashboardMetricsResponseProjectUsageItem']):
        user_usage (list['DashboardMetricsResponseUserUsageItem']):
        token_breakdown (DashboardMetricsResponseTokenbreakdown):
        tool_usage (list['DashboardMetricsResponseToolUsageItem']):
    """

    summary: "DashboardMetricsResponseSummary"
    trace_count: int
    llm_usage: "DashboardMetricsResponseLlmusage"
    project_usage: list["DashboardMetricsResponseProjectUsageItem"]
    user_usage: list["DashboardMetricsResponseUserUsageItem"]
    token_breakdown: "DashboardMetricsResponseTokenbreakdown"
    tool_usage: list["DashboardMetricsResponseToolUsageItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        summary = self.summary.to_dict()

        trace_count = self.trace_count

        llm_usage = self.llm_usage.to_dict()

        project_usage = []
        for project_usage_item_data in self.project_usage:
            project_usage_item = project_usage_item_data.to_dict()
            project_usage.append(project_usage_item)

        user_usage = []
        for user_usage_item_data in self.user_usage:
            user_usage_item = user_usage_item_data.to_dict()
            user_usage.append(user_usage_item)

        token_breakdown = self.token_breakdown.to_dict()

        tool_usage = []
        for tool_usage_item_data in self.tool_usage:
            tool_usage_item = tool_usage_item_data.to_dict()
            tool_usage.append(tool_usage_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "summary": summary,
                "traceCount": trace_count,
                "llmUsage": llm_usage,
                "projectUsage": project_usage,
                "userUsage": user_usage,
                "tokenBreakdown": token_breakdown,
                "toolUsage": tool_usage,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dashboard_metrics_response_llmusage import (
            DashboardMetricsResponseLlmusage,
        )
        from ..models.dashboard_metrics_response_project_usage_item import (
            DashboardMetricsResponseProjectUsageItem,
        )
        from ..models.dashboard_metrics_response_summary import (
            DashboardMetricsResponseSummary,
        )
        from ..models.dashboard_metrics_response_tokenbreakdown import (
            DashboardMetricsResponseTokenbreakdown,
        )
        from ..models.dashboard_metrics_response_tool_usage_item import (
            DashboardMetricsResponseToolUsageItem,
        )
        from ..models.dashboard_metrics_response_user_usage_item import (
            DashboardMetricsResponseUserUsageItem,
        )

        d = dict(src_dict)
        summary = DashboardMetricsResponseSummary.from_dict(d.pop("summary"))

        trace_count = d.pop("traceCount")

        llm_usage = DashboardMetricsResponseLlmusage.from_dict(d.pop("llmUsage"))

        project_usage = []
        _project_usage = d.pop("projectUsage")
        for project_usage_item_data in _project_usage:
            project_usage_item = DashboardMetricsResponseProjectUsageItem.from_dict(
                project_usage_item_data
            )

            project_usage.append(project_usage_item)

        user_usage = []
        _user_usage = d.pop("userUsage")
        for user_usage_item_data in _user_usage:
            user_usage_item = DashboardMetricsResponseUserUsageItem.from_dict(
                user_usage_item_data
            )

            user_usage.append(user_usage_item)

        token_breakdown = DashboardMetricsResponseTokenbreakdown.from_dict(
            d.pop("tokenBreakdown")
        )

        tool_usage = []
        _tool_usage = d.pop("toolUsage")
        for tool_usage_item_data in _tool_usage:
            tool_usage_item = DashboardMetricsResponseToolUsageItem.from_dict(
                tool_usage_item_data
            )

            tool_usage.append(tool_usage_item)

        dashboard_metrics_response = cls(
            summary=summary,
            trace_count=trace_count,
            llm_usage=llm_usage,
            project_usage=project_usage,
            user_usage=user_usage,
            token_breakdown=token_breakdown,
            tool_usage=tool_usage,
        )

        dashboard_metrics_response.additional_properties = d
        return dashboard_metrics_response

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
