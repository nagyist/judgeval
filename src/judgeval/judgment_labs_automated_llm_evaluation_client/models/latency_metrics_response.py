from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.latency_metrics_response_llm_latency_item import (
        LatencyMetricsResponseLlmLatencyItem,
    )
    from ..models.latency_metrics_response_llm_percentiles import (
        LatencyMetricsResponseLlmPercentiles,
    )
    from ..models.latency_metrics_response_tool_latency_item import (
        LatencyMetricsResponseToolLatencyItem,
    )
    from ..models.latency_metrics_response_tool_percentiles import (
        LatencyMetricsResponseToolPercentiles,
    )
    from ..models.latency_metrics_response_trace_latency_item import (
        LatencyMetricsResponseTraceLatencyItem,
    )
    from ..models.latency_metrics_response_trace_percentiles import (
        LatencyMetricsResponseTracePercentiles,
    )


T = TypeVar("T", bound="LatencyMetricsResponse")


@_attrs_define
class LatencyMetricsResponse:
    """
    Attributes:
        trace_latency (list['LatencyMetricsResponseTraceLatencyItem']):
        llm_latency (list['LatencyMetricsResponseLlmLatencyItem']):
        tool_latency (list['LatencyMetricsResponseToolLatencyItem']):
        trace_percentiles (LatencyMetricsResponseTracePercentiles):
        llm_percentiles (LatencyMetricsResponseLlmPercentiles):
        tool_percentiles (LatencyMetricsResponseToolPercentiles):
    """

    trace_latency: list["LatencyMetricsResponseTraceLatencyItem"]
    llm_latency: list["LatencyMetricsResponseLlmLatencyItem"]
    tool_latency: list["LatencyMetricsResponseToolLatencyItem"]
    trace_percentiles: "LatencyMetricsResponseTracePercentiles"
    llm_percentiles: "LatencyMetricsResponseLlmPercentiles"
    tool_percentiles: "LatencyMetricsResponseToolPercentiles"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        trace_latency = []
        for trace_latency_item_data in self.trace_latency:
            trace_latency_item = trace_latency_item_data.to_dict()
            trace_latency.append(trace_latency_item)

        llm_latency = []
        for llm_latency_item_data in self.llm_latency:
            llm_latency_item = llm_latency_item_data.to_dict()
            llm_latency.append(llm_latency_item)

        tool_latency = []
        for tool_latency_item_data in self.tool_latency:
            tool_latency_item = tool_latency_item_data.to_dict()
            tool_latency.append(tool_latency_item)

        trace_percentiles = self.trace_percentiles.to_dict()

        llm_percentiles = self.llm_percentiles.to_dict()

        tool_percentiles = self.tool_percentiles.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "trace_latency": trace_latency,
                "llm_latency": llm_latency,
                "tool_latency": tool_latency,
                "trace_percentiles": trace_percentiles,
                "llm_percentiles": llm_percentiles,
                "tool_percentiles": tool_percentiles,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.latency_metrics_response_llm_latency_item import (
            LatencyMetricsResponseLlmLatencyItem,
        )
        from ..models.latency_metrics_response_llm_percentiles import (
            LatencyMetricsResponseLlmPercentiles,
        )
        from ..models.latency_metrics_response_tool_latency_item import (
            LatencyMetricsResponseToolLatencyItem,
        )
        from ..models.latency_metrics_response_tool_percentiles import (
            LatencyMetricsResponseToolPercentiles,
        )
        from ..models.latency_metrics_response_trace_latency_item import (
            LatencyMetricsResponseTraceLatencyItem,
        )
        from ..models.latency_metrics_response_trace_percentiles import (
            LatencyMetricsResponseTracePercentiles,
        )

        d = dict(src_dict)
        trace_latency = []
        _trace_latency = d.pop("trace_latency")
        for trace_latency_item_data in _trace_latency:
            trace_latency_item = LatencyMetricsResponseTraceLatencyItem.from_dict(
                trace_latency_item_data
            )

            trace_latency.append(trace_latency_item)

        llm_latency = []
        _llm_latency = d.pop("llm_latency")
        for llm_latency_item_data in _llm_latency:
            llm_latency_item = LatencyMetricsResponseLlmLatencyItem.from_dict(
                llm_latency_item_data
            )

            llm_latency.append(llm_latency_item)

        tool_latency = []
        _tool_latency = d.pop("tool_latency")
        for tool_latency_item_data in _tool_latency:
            tool_latency_item = LatencyMetricsResponseToolLatencyItem.from_dict(
                tool_latency_item_data
            )

            tool_latency.append(tool_latency_item)

        trace_percentiles = LatencyMetricsResponseTracePercentiles.from_dict(
            d.pop("trace_percentiles")
        )

        llm_percentiles = LatencyMetricsResponseLlmPercentiles.from_dict(
            d.pop("llm_percentiles")
        )

        tool_percentiles = LatencyMetricsResponseToolPercentiles.from_dict(
            d.pop("tool_percentiles")
        )

        latency_metrics_response = cls(
            trace_latency=trace_latency,
            llm_latency=llm_latency,
            tool_latency=tool_latency,
            trace_percentiles=trace_percentiles,
            llm_percentiles=llm_percentiles,
            tool_percentiles=tool_percentiles,
        )

        latency_metrics_response.additional_properties = d
        return latency_metrics_response

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
