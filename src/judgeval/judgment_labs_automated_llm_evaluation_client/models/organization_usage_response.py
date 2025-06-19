from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.subscription_tier import SubscriptionTier
from ..types import UNSET, Unset

T = TypeVar("T", bound="OrganizationUsageResponse")


@_attrs_define
class OrganizationUsageResponse:
    """
    Attributes:
        judgee_limit (int):
        trace_limit (int):
        judgee_used (int):
        trace_used (int):
        judgee_remaining (int):
        trace_remaining (int):
        on_demand_judgee_used (int):
        on_demand_judgee_limit (int):
        on_demand_judgee_remaining (int):
        on_demand_trace_used (int):
        on_demand_trace_limit (int):
        on_demand_trace_remaining (int):
        subscription_tier (SubscriptionTier):
        usage_based_enabled (Union[Unset, bool]):  Default: False.
        payg_cost_evals_cents (Union[Unset, int]):  Default: 0.
        payg_cost_traces_cents (Union[Unset, int]):  Default: 0.
    """

    judgee_limit: int
    trace_limit: int
    judgee_used: int
    trace_used: int
    judgee_remaining: int
    trace_remaining: int
    on_demand_judgee_used: int
    on_demand_judgee_limit: int
    on_demand_judgee_remaining: int
    on_demand_trace_used: int
    on_demand_trace_limit: int
    on_demand_trace_remaining: int
    subscription_tier: SubscriptionTier
    usage_based_enabled: Union[Unset, bool] = False
    payg_cost_evals_cents: Union[Unset, int] = 0
    payg_cost_traces_cents: Union[Unset, int] = 0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        judgee_limit = self.judgee_limit

        trace_limit = self.trace_limit

        judgee_used = self.judgee_used

        trace_used = self.trace_used

        judgee_remaining = self.judgee_remaining

        trace_remaining = self.trace_remaining

        on_demand_judgee_used = self.on_demand_judgee_used

        on_demand_judgee_limit = self.on_demand_judgee_limit

        on_demand_judgee_remaining = self.on_demand_judgee_remaining

        on_demand_trace_used = self.on_demand_trace_used

        on_demand_trace_limit = self.on_demand_trace_limit

        on_demand_trace_remaining = self.on_demand_trace_remaining

        subscription_tier = self.subscription_tier.value

        usage_based_enabled = self.usage_based_enabled

        payg_cost_evals_cents = self.payg_cost_evals_cents

        payg_cost_traces_cents = self.payg_cost_traces_cents

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "judgee_limit": judgee_limit,
                "trace_limit": trace_limit,
                "judgee_used": judgee_used,
                "trace_used": trace_used,
                "judgee_remaining": judgee_remaining,
                "trace_remaining": trace_remaining,
                "on_demand_judgee_used": on_demand_judgee_used,
                "on_demand_judgee_limit": on_demand_judgee_limit,
                "on_demand_judgee_remaining": on_demand_judgee_remaining,
                "on_demand_trace_used": on_demand_trace_used,
                "on_demand_trace_limit": on_demand_trace_limit,
                "on_demand_trace_remaining": on_demand_trace_remaining,
                "subscription_tier": subscription_tier,
            }
        )
        if usage_based_enabled is not UNSET:
            field_dict["usage_based_enabled"] = usage_based_enabled
        if payg_cost_evals_cents is not UNSET:
            field_dict["payg_cost_evals_cents"] = payg_cost_evals_cents
        if payg_cost_traces_cents is not UNSET:
            field_dict["payg_cost_traces_cents"] = payg_cost_traces_cents

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        judgee_limit = d.pop("judgee_limit")

        trace_limit = d.pop("trace_limit")

        judgee_used = d.pop("judgee_used")

        trace_used = d.pop("trace_used")

        judgee_remaining = d.pop("judgee_remaining")

        trace_remaining = d.pop("trace_remaining")

        on_demand_judgee_used = d.pop("on_demand_judgee_used")

        on_demand_judgee_limit = d.pop("on_demand_judgee_limit")

        on_demand_judgee_remaining = d.pop("on_demand_judgee_remaining")

        on_demand_trace_used = d.pop("on_demand_trace_used")

        on_demand_trace_limit = d.pop("on_demand_trace_limit")

        on_demand_trace_remaining = d.pop("on_demand_trace_remaining")

        subscription_tier = SubscriptionTier(d.pop("subscription_tier"))

        usage_based_enabled = d.pop("usage_based_enabled", UNSET)

        payg_cost_evals_cents = d.pop("payg_cost_evals_cents", UNSET)

        payg_cost_traces_cents = d.pop("payg_cost_traces_cents", UNSET)

        organization_usage_response = cls(
            judgee_limit=judgee_limit,
            trace_limit=trace_limit,
            judgee_used=judgee_used,
            trace_used=trace_used,
            judgee_remaining=judgee_remaining,
            trace_remaining=trace_remaining,
            on_demand_judgee_used=on_demand_judgee_used,
            on_demand_judgee_limit=on_demand_judgee_limit,
            on_demand_judgee_remaining=on_demand_judgee_remaining,
            on_demand_trace_used=on_demand_trace_used,
            on_demand_trace_limit=on_demand_trace_limit,
            on_demand_trace_remaining=on_demand_trace_remaining,
            subscription_tier=subscription_tier,
            usage_based_enabled=usage_based_enabled,
            payg_cost_evals_cents=payg_cost_evals_cents,
            payg_cost_traces_cents=payg_cost_traces_cents,
        )

        organization_usage_response.additional_properties = d
        return organization_usage_response

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
