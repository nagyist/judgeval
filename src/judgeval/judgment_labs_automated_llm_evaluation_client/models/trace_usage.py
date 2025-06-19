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

T = TypeVar("T", bound="TraceUsage")


@_attrs_define
class TraceUsage:
    """
    Attributes:
        prompt_tokens (Union[None, Unset, int]):
        completion_tokens (Union[None, Unset, int]):
        total_tokens (Union[None, Unset, int]):
        prompt_tokens_cost_usd (Union[None, Unset, float]):
        completion_tokens_cost_usd (Union[None, Unset, float]):
        total_cost_usd (Union[None, Unset, float]):
        model_name (Union[None, Unset, str]):
    """

    prompt_tokens: Union[None, Unset, int] = UNSET
    completion_tokens: Union[None, Unset, int] = UNSET
    total_tokens: Union[None, Unset, int] = UNSET
    prompt_tokens_cost_usd: Union[None, Unset, float] = UNSET
    completion_tokens_cost_usd: Union[None, Unset, float] = UNSET
    total_cost_usd: Union[None, Unset, float] = UNSET
    model_name: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        prompt_tokens: Union[None, Unset, int]
        if isinstance(self.prompt_tokens, Unset):
            prompt_tokens = UNSET
        else:
            prompt_tokens = self.prompt_tokens

        completion_tokens: Union[None, Unset, int]
        if isinstance(self.completion_tokens, Unset):
            completion_tokens = UNSET
        else:
            completion_tokens = self.completion_tokens

        total_tokens: Union[None, Unset, int]
        if isinstance(self.total_tokens, Unset):
            total_tokens = UNSET
        else:
            total_tokens = self.total_tokens

        prompt_tokens_cost_usd: Union[None, Unset, float]
        if isinstance(self.prompt_tokens_cost_usd, Unset):
            prompt_tokens_cost_usd = UNSET
        else:
            prompt_tokens_cost_usd = self.prompt_tokens_cost_usd

        completion_tokens_cost_usd: Union[None, Unset, float]
        if isinstance(self.completion_tokens_cost_usd, Unset):
            completion_tokens_cost_usd = UNSET
        else:
            completion_tokens_cost_usd = self.completion_tokens_cost_usd

        total_cost_usd: Union[None, Unset, float]
        if isinstance(self.total_cost_usd, Unset):
            total_cost_usd = UNSET
        else:
            total_cost_usd = self.total_cost_usd

        model_name: Union[None, Unset, str]
        if isinstance(self.model_name, Unset):
            model_name = UNSET
        else:
            model_name = self.model_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if prompt_tokens is not UNSET:
            field_dict["prompt_tokens"] = prompt_tokens
        if completion_tokens is not UNSET:
            field_dict["completion_tokens"] = completion_tokens
        if total_tokens is not UNSET:
            field_dict["total_tokens"] = total_tokens
        if prompt_tokens_cost_usd is not UNSET:
            field_dict["prompt_tokens_cost_usd"] = prompt_tokens_cost_usd
        if completion_tokens_cost_usd is not UNSET:
            field_dict["completion_tokens_cost_usd"] = completion_tokens_cost_usd
        if total_cost_usd is not UNSET:
            field_dict["total_cost_usd"] = total_cost_usd
        if model_name is not UNSET:
            field_dict["model_name"] = model_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_prompt_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        prompt_tokens = _parse_prompt_tokens(d.pop("prompt_tokens", UNSET))

        def _parse_completion_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        completion_tokens = _parse_completion_tokens(d.pop("completion_tokens", UNSET))

        def _parse_total_tokens(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        total_tokens = _parse_total_tokens(d.pop("total_tokens", UNSET))

        def _parse_prompt_tokens_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        prompt_tokens_cost_usd = _parse_prompt_tokens_cost_usd(
            d.pop("prompt_tokens_cost_usd", UNSET)
        )

        def _parse_completion_tokens_cost_usd(
            data: object,
        ) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        completion_tokens_cost_usd = _parse_completion_tokens_cost_usd(
            d.pop("completion_tokens_cost_usd", UNSET)
        )

        def _parse_total_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        total_cost_usd = _parse_total_cost_usd(d.pop("total_cost_usd", UNSET))

        def _parse_model_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        model_name = _parse_model_name(d.pop("model_name", UNSET))

        trace_usage = cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_cost_usd=prompt_tokens_cost_usd,
            completion_tokens_cost_usd=completion_tokens_cost_usd,
            total_cost_usd=total_cost_usd,
            model_name=model_name,
        )

        trace_usage.additional_properties = d
        return trace_usage

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
