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
    from ..models.tool_action_dependencies_type_0_item import (
        ToolActionDependenciesType0Item,
    )
    from ..models.tool_parameters_type_0 import ToolParametersType0
    from ..models.tool_result_dependencies_type_0_item import (
        ToolResultDependenciesType0Item,
    )


T = TypeVar("T", bound="Tool")


@_attrs_define
class Tool:
    """
    Attributes:
        tool_name (str):
        parameters (Union['ToolParametersType0', None, Unset]):
        agent_name (Union[None, Unset, str]):
        result_dependencies (Union[None, Unset, list['ToolResultDependenciesType0Item']]):
        action_dependencies (Union[None, Unset, list['ToolActionDependenciesType0Item']]):
        require_all (Union[None, Unset, bool]):
    """

    tool_name: str
    parameters: Union["ToolParametersType0", None, Unset] = UNSET
    agent_name: Union[None, Unset, str] = UNSET
    result_dependencies: Union[None, Unset, list["ToolResultDependenciesType0Item"]] = (
        UNSET
    )
    action_dependencies: Union[None, Unset, list["ToolActionDependenciesType0Item"]] = (
        UNSET
    )
    require_all: Union[None, Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.tool_parameters_type_0 import ToolParametersType0

        tool_name = self.tool_name

        parameters: Union[None, Unset, dict[str, Any]]
        if isinstance(self.parameters, Unset):
            parameters = UNSET
        elif isinstance(self.parameters, ToolParametersType0):
            parameters = self.parameters.to_dict()
        else:
            parameters = self.parameters

        agent_name: Union[None, Unset, str]
        if isinstance(self.agent_name, Unset):
            agent_name = UNSET
        else:
            agent_name = self.agent_name

        result_dependencies: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.result_dependencies, Unset):
            result_dependencies = UNSET
        elif isinstance(self.result_dependencies, list):
            result_dependencies = []
            for result_dependencies_type_0_item_data in self.result_dependencies:
                result_dependencies_type_0_item = (
                    result_dependencies_type_0_item_data.to_dict()
                )
                result_dependencies.append(result_dependencies_type_0_item)

        else:
            result_dependencies = self.result_dependencies

        action_dependencies: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.action_dependencies, Unset):
            action_dependencies = UNSET
        elif isinstance(self.action_dependencies, list):
            action_dependencies = []
            for action_dependencies_type_0_item_data in self.action_dependencies:
                action_dependencies_type_0_item = (
                    action_dependencies_type_0_item_data.to_dict()
                )
                action_dependencies.append(action_dependencies_type_0_item)

        else:
            action_dependencies = self.action_dependencies

        require_all: Union[None, Unset, bool]
        if isinstance(self.require_all, Unset):
            require_all = UNSET
        else:
            require_all = self.require_all

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "tool_name": tool_name,
            }
        )
        if parameters is not UNSET:
            field_dict["parameters"] = parameters
        if agent_name is not UNSET:
            field_dict["agent_name"] = agent_name
        if result_dependencies is not UNSET:
            field_dict["result_dependencies"] = result_dependencies
        if action_dependencies is not UNSET:
            field_dict["action_dependencies"] = action_dependencies
        if require_all is not UNSET:
            field_dict["require_all"] = require_all

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tool_action_dependencies_type_0_item import (
            ToolActionDependenciesType0Item,
        )
        from ..models.tool_parameters_type_0 import ToolParametersType0
        from ..models.tool_result_dependencies_type_0_item import (
            ToolResultDependenciesType0Item,
        )

        d = dict(src_dict)
        tool_name = d.pop("tool_name")

        def _parse_parameters(
            data: object,
        ) -> Union["ToolParametersType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                parameters_type_0 = ToolParametersType0.from_dict(data)

                return parameters_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ToolParametersType0", None, Unset], data)

        parameters = _parse_parameters(d.pop("parameters", UNSET))

        def _parse_agent_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        agent_name = _parse_agent_name(d.pop("agent_name", UNSET))

        def _parse_result_dependencies(
            data: object,
        ) -> Union[None, Unset, list["ToolResultDependenciesType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                result_dependencies_type_0 = []
                _result_dependencies_type_0 = data
                for result_dependencies_type_0_item_data in _result_dependencies_type_0:
                    result_dependencies_type_0_item = (
                        ToolResultDependenciesType0Item.from_dict(
                            result_dependencies_type_0_item_data
                        )
                    )

                    result_dependencies_type_0.append(result_dependencies_type_0_item)

                return result_dependencies_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union[None, Unset, list["ToolResultDependenciesType0Item"]], data
            )

        result_dependencies = _parse_result_dependencies(
            d.pop("result_dependencies", UNSET)
        )

        def _parse_action_dependencies(
            data: object,
        ) -> Union[None, Unset, list["ToolActionDependenciesType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                action_dependencies_type_0 = []
                _action_dependencies_type_0 = data
                for action_dependencies_type_0_item_data in _action_dependencies_type_0:
                    action_dependencies_type_0_item = (
                        ToolActionDependenciesType0Item.from_dict(
                            action_dependencies_type_0_item_data
                        )
                    )

                    action_dependencies_type_0.append(action_dependencies_type_0_item)

                return action_dependencies_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union[None, Unset, list["ToolActionDependenciesType0Item"]], data
            )

        action_dependencies = _parse_action_dependencies(
            d.pop("action_dependencies", UNSET)
        )

        def _parse_require_all(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        require_all = _parse_require_all(d.pop("require_all", UNSET))

        tool = cls(
            tool_name=tool_name,
            parameters=parameters,
            agent_name=agent_name,
            result_dependencies=result_dependencies,
            action_dependencies=action_dependencies,
            require_all=require_all,
        )

        tool.additional_properties = d
        return tool

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
