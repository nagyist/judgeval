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
    from ..models.custom_example_actual_output_type_0 import (
        CustomExampleActualOutputType0,
    )
    from ..models.custom_example_additional_metadata_type_0 import (
        CustomExampleAdditionalMetadataType0,
    )
    from ..models.custom_example_expected_output_type_0 import (
        CustomExampleExpectedOutputType0,
    )
    from ..models.custom_example_input_type_0 import CustomExampleInputType0


T = TypeVar("T", bound="CustomExample")


@_attrs_define
class CustomExample:
    """
    Attributes:
        input_ (Union['CustomExampleInputType0', None, Unset]):
        actual_output (Union['CustomExampleActualOutputType0', None, Unset]):
        expected_output (Union['CustomExampleExpectedOutputType0', None, Unset]):
        context (Union[None, Unset, list[str]]):
        retrieval_context (Union[None, Unset, list[str]]):
        additional_metadata (Union['CustomExampleAdditionalMetadataType0', None, Unset]):
        tools_called (Union[None, Unset, list[str]]):
        expected_tools (Union[None, Unset, list[str]]):
        name (Union[None, Unset, str]):
        example_id (Union[Unset, str]):
        example_index (Union[None, Unset, int]):
        timestamp (Union[None, Unset, str]):
        trace_id (Union[None, Unset, str]):
    """

    input_: Union["CustomExampleInputType0", None, Unset] = UNSET
    actual_output: Union["CustomExampleActualOutputType0", None, Unset] = UNSET
    expected_output: Union["CustomExampleExpectedOutputType0", None, Unset] = UNSET
    context: Union[None, Unset, list[str]] = UNSET
    retrieval_context: Union[None, Unset, list[str]] = UNSET
    additional_metadata: Union["CustomExampleAdditionalMetadataType0", None, Unset] = (
        UNSET
    )
    tools_called: Union[None, Unset, list[str]] = UNSET
    expected_tools: Union[None, Unset, list[str]] = UNSET
    name: Union[None, Unset, str] = UNSET
    example_id: Union[Unset, str] = UNSET
    example_index: Union[None, Unset, int] = UNSET
    timestamp: Union[None, Unset, str] = UNSET
    trace_id: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.custom_example_actual_output_type_0 import (
            CustomExampleActualOutputType0,
        )
        from ..models.custom_example_additional_metadata_type_0 import (
            CustomExampleAdditionalMetadataType0,
        )
        from ..models.custom_example_expected_output_type_0 import (
            CustomExampleExpectedOutputType0,
        )
        from ..models.custom_example_input_type_0 import CustomExampleInputType0

        input_: Union[None, Unset, dict[str, Any]]
        if isinstance(self.input_, Unset):
            input_ = UNSET
        elif isinstance(self.input_, CustomExampleInputType0):
            input_ = self.input_.to_dict()
        else:
            input_ = self.input_

        actual_output: Union[None, Unset, dict[str, Any]]
        if isinstance(self.actual_output, Unset):
            actual_output = UNSET
        elif isinstance(self.actual_output, CustomExampleActualOutputType0):
            actual_output = self.actual_output.to_dict()
        else:
            actual_output = self.actual_output

        expected_output: Union[None, Unset, dict[str, Any]]
        if isinstance(self.expected_output, Unset):
            expected_output = UNSET
        elif isinstance(self.expected_output, CustomExampleExpectedOutputType0):
            expected_output = self.expected_output.to_dict()
        else:
            expected_output = self.expected_output

        context: Union[None, Unset, list[str]]
        if isinstance(self.context, Unset):
            context = UNSET
        elif isinstance(self.context, list):
            context = self.context

        else:
            context = self.context

        retrieval_context: Union[None, Unset, list[str]]
        if isinstance(self.retrieval_context, Unset):
            retrieval_context = UNSET
        elif isinstance(self.retrieval_context, list):
            retrieval_context = self.retrieval_context

        else:
            retrieval_context = self.retrieval_context

        additional_metadata: Union[None, Unset, dict[str, Any]]
        if isinstance(self.additional_metadata, Unset):
            additional_metadata = UNSET
        elif isinstance(self.additional_metadata, CustomExampleAdditionalMetadataType0):
            additional_metadata = self.additional_metadata.to_dict()
        else:
            additional_metadata = self.additional_metadata

        tools_called: Union[None, Unset, list[str]]
        if isinstance(self.tools_called, Unset):
            tools_called = UNSET
        elif isinstance(self.tools_called, list):
            tools_called = self.tools_called

        else:
            tools_called = self.tools_called

        expected_tools: Union[None, Unset, list[str]]
        if isinstance(self.expected_tools, Unset):
            expected_tools = UNSET
        elif isinstance(self.expected_tools, list):
            expected_tools = self.expected_tools

        else:
            expected_tools = self.expected_tools

        name: Union[None, Unset, str]
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        example_id = self.example_id

        example_index: Union[None, Unset, int]
        if isinstance(self.example_index, Unset):
            example_index = UNSET
        else:
            example_index = self.example_index

        timestamp: Union[None, Unset, str]
        if isinstance(self.timestamp, Unset):
            timestamp = UNSET
        else:
            timestamp = self.timestamp

        trace_id: Union[None, Unset, str]
        if isinstance(self.trace_id, Unset):
            trace_id = UNSET
        else:
            trace_id = self.trace_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if input_ is not UNSET:
            field_dict["input"] = input_
        if actual_output is not UNSET:
            field_dict["actual_output"] = actual_output
        if expected_output is not UNSET:
            field_dict["expected_output"] = expected_output
        if context is not UNSET:
            field_dict["context"] = context
        if retrieval_context is not UNSET:
            field_dict["retrieval_context"] = retrieval_context
        if additional_metadata is not UNSET:
            field_dict["additional_metadata"] = additional_metadata
        if tools_called is not UNSET:
            field_dict["tools_called"] = tools_called
        if expected_tools is not UNSET:
            field_dict["expected_tools"] = expected_tools
        if name is not UNSET:
            field_dict["name"] = name
        if example_id is not UNSET:
            field_dict["example_id"] = example_id
        if example_index is not UNSET:
            field_dict["example_index"] = example_index
        if timestamp is not UNSET:
            field_dict["timestamp"] = timestamp
        if trace_id is not UNSET:
            field_dict["trace_id"] = trace_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.custom_example_actual_output_type_0 import (
            CustomExampleActualOutputType0,
        )
        from ..models.custom_example_additional_metadata_type_0 import (
            CustomExampleAdditionalMetadataType0,
        )
        from ..models.custom_example_expected_output_type_0 import (
            CustomExampleExpectedOutputType0,
        )
        from ..models.custom_example_input_type_0 import CustomExampleInputType0

        d = dict(src_dict)

        def _parse_input_(
            data: object,
        ) -> Union["CustomExampleInputType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_type_0 = CustomExampleInputType0.from_dict(data)

                return input_type_0
            except:  # noqa: E722
                pass
            return cast(Union["CustomExampleInputType0", None, Unset], data)

        input_ = _parse_input_(d.pop("input", UNSET))

        def _parse_actual_output(
            data: object,
        ) -> Union["CustomExampleActualOutputType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                actual_output_type_0 = CustomExampleActualOutputType0.from_dict(data)

                return actual_output_type_0
            except:  # noqa: E722
                pass
            return cast(Union["CustomExampleActualOutputType0", None, Unset], data)

        actual_output = _parse_actual_output(d.pop("actual_output", UNSET))

        def _parse_expected_output(
            data: object,
        ) -> Union["CustomExampleExpectedOutputType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                expected_output_type_0 = CustomExampleExpectedOutputType0.from_dict(
                    data
                )

                return expected_output_type_0
            except:  # noqa: E722
                pass
            return cast(Union["CustomExampleExpectedOutputType0", None, Unset], data)

        expected_output = _parse_expected_output(d.pop("expected_output", UNSET))

        def _parse_context(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                context_type_0 = cast(list[str], data)

                return context_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        context = _parse_context(d.pop("context", UNSET))

        def _parse_retrieval_context(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                retrieval_context_type_0 = cast(list[str], data)

                return retrieval_context_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        retrieval_context = _parse_retrieval_context(d.pop("retrieval_context", UNSET))

        def _parse_additional_metadata(
            data: object,
        ) -> Union["CustomExampleAdditionalMetadataType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                additional_metadata_type_0 = (
                    CustomExampleAdditionalMetadataType0.from_dict(data)
                )

                return additional_metadata_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union["CustomExampleAdditionalMetadataType0", None, Unset], data
            )

        additional_metadata = _parse_additional_metadata(
            d.pop("additional_metadata", UNSET)
        )

        def _parse_tools_called(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_called_type_0 = cast(list[str], data)

                return tools_called_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        tools_called = _parse_tools_called(d.pop("tools_called", UNSET))

        def _parse_expected_tools(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                expected_tools_type_0 = cast(list[str], data)

                return expected_tools_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        expected_tools = _parse_expected_tools(d.pop("expected_tools", UNSET))

        def _parse_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        name = _parse_name(d.pop("name", UNSET))

        example_id = d.pop("example_id", UNSET)

        def _parse_example_index(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        example_index = _parse_example_index(d.pop("example_index", UNSET))

        def _parse_timestamp(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        timestamp = _parse_timestamp(d.pop("timestamp", UNSET))

        def _parse_trace_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        trace_id = _parse_trace_id(d.pop("trace_id", UNSET))

        custom_example = cls(
            input_=input_,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            additional_metadata=additional_metadata,
            tools_called=tools_called,
            expected_tools=expected_tools,
            name=name,
            example_id=example_id,
            example_index=example_index,
            timestamp=timestamp,
            trace_id=trace_id,
        )

        custom_example.additional_properties = d
        return custom_example

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
