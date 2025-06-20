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
    from ..models.example_additional_metadata_type_0 import (
        ExampleAdditionalMetadataType0,
    )
    from ..models.example_input_type_1 import ExampleInputType1
    from ..models.tool import Tool


T = TypeVar("T", bound="Example")


@_attrs_define
class Example:
    """Based on the examples table in the database

    Attributes:
        example_id (str):
        input_ (Union['ExampleInputType1', None, Unset, str]):
        actual_output (Union[None, Unset, list[str], str]):
        expected_output (Union[None, Unset, list[str], str]):
        context (Union[None, Unset, list[str]]):
        retrieval_context (Union[None, Unset, list[str]]):
        additional_metadata (Union['ExampleAdditionalMetadataType0', None, Unset]):
        tools_called (Union[None, Unset, list[str]]):
        expected_tools (Union[None, Unset, list['Tool']]):
        name (Union[None, Unset, str]):
        created_at (Union[None, Unset, str]):
        dataset_id (Union[None, Unset, str]):
        trace_span_id (Union[None, Unset, str]):
    """

    example_id: str
    input_: Union["ExampleInputType1", None, Unset, str] = UNSET
    actual_output: Union[None, Unset, list[str], str] = UNSET
    expected_output: Union[None, Unset, list[str], str] = UNSET
    context: Union[None, Unset, list[str]] = UNSET
    retrieval_context: Union[None, Unset, list[str]] = UNSET
    additional_metadata: Union["ExampleAdditionalMetadataType0", None, Unset] = UNSET
    tools_called: Union[None, Unset, list[str]] = UNSET
    expected_tools: Union[None, Unset, list["Tool"]] = UNSET
    name: Union[None, Unset, str] = UNSET
    created_at: Union[None, Unset, str] = UNSET
    dataset_id: Union[None, Unset, str] = UNSET
    trace_span_id: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.example_additional_metadata_type_0 import (
            ExampleAdditionalMetadataType0,
        )
        from ..models.example_input_type_1 import ExampleInputType1

        example_id = self.example_id

        input_: Union[None, Unset, dict[str, Any], str]
        if isinstance(self.input_, Unset):
            input_ = UNSET
        elif isinstance(self.input_, ExampleInputType1):
            input_ = self.input_.to_dict()
        else:
            input_ = self.input_

        actual_output: Union[None, Unset, list[str], str]
        if isinstance(self.actual_output, Unset):
            actual_output = UNSET
        elif isinstance(self.actual_output, list):
            actual_output = self.actual_output

        else:
            actual_output = self.actual_output

        expected_output: Union[None, Unset, list[str], str]
        if isinstance(self.expected_output, Unset):
            expected_output = UNSET
        elif isinstance(self.expected_output, list):
            expected_output = self.expected_output

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
        elif isinstance(self.additional_metadata, ExampleAdditionalMetadataType0):
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

        expected_tools: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.expected_tools, Unset):
            expected_tools = UNSET
        elif isinstance(self.expected_tools, list):
            expected_tools = []
            for expected_tools_type_0_item_data in self.expected_tools:
                expected_tools_type_0_item = expected_tools_type_0_item_data.to_dict()
                expected_tools.append(expected_tools_type_0_item)

        else:
            expected_tools = self.expected_tools

        name: Union[None, Unset, str]
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        created_at: Union[None, Unset, str]
        if isinstance(self.created_at, Unset):
            created_at = UNSET
        else:
            created_at = self.created_at

        dataset_id: Union[None, Unset, str]
        if isinstance(self.dataset_id, Unset):
            dataset_id = UNSET
        else:
            dataset_id = self.dataset_id

        trace_span_id: Union[None, Unset, str]
        if isinstance(self.trace_span_id, Unset):
            trace_span_id = UNSET
        else:
            trace_span_id = self.trace_span_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "example_id": example_id,
            }
        )
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
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if dataset_id is not UNSET:
            field_dict["dataset_id"] = dataset_id
        if trace_span_id is not UNSET:
            field_dict["trace_span_id"] = trace_span_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.example_additional_metadata_type_0 import (
            ExampleAdditionalMetadataType0,
        )
        from ..models.example_input_type_1 import ExampleInputType1
        from ..models.tool import Tool

        d = dict(src_dict)
        example_id = d.pop("example_id")

        def _parse_input_(data: object) -> Union["ExampleInputType1", None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_type_1 = ExampleInputType1.from_dict(data)

                return input_type_1
            except:  # noqa: E722
                pass
            return cast(Union["ExampleInputType1", None, Unset, str], data)

        input_ = _parse_input_(d.pop("input", UNSET))

        def _parse_actual_output(data: object) -> Union[None, Unset, list[str], str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                actual_output_type_1 = cast(list[str], data)

                return actual_output_type_1
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str], str], data)

        actual_output = _parse_actual_output(d.pop("actual_output", UNSET))

        def _parse_expected_output(data: object) -> Union[None, Unset, list[str], str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                expected_output_type_1 = cast(list[str], data)

                return expected_output_type_1
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str], str], data)

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
        ) -> Union["ExampleAdditionalMetadataType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                additional_metadata_type_0 = ExampleAdditionalMetadataType0.from_dict(
                    data
                )

                return additional_metadata_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ExampleAdditionalMetadataType0", None, Unset], data)

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

        def _parse_expected_tools(data: object) -> Union[None, Unset, list["Tool"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                expected_tools_type_0 = []
                _expected_tools_type_0 = data
                for expected_tools_type_0_item_data in _expected_tools_type_0:
                    expected_tools_type_0_item = Tool.from_dict(
                        expected_tools_type_0_item_data
                    )

                    expected_tools_type_0.append(expected_tools_type_0_item)

                return expected_tools_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["Tool"]], data)

        expected_tools = _parse_expected_tools(d.pop("expected_tools", UNSET))

        def _parse_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_created_at(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        created_at = _parse_created_at(d.pop("created_at", UNSET))

        def _parse_dataset_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        dataset_id = _parse_dataset_id(d.pop("dataset_id", UNSET))

        def _parse_trace_span_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        trace_span_id = _parse_trace_span_id(d.pop("trace_span_id", UNSET))

        example = cls(
            example_id=example_id,
            input_=input_,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            additional_metadata=additional_metadata,
            tools_called=tools_called,
            expected_tools=expected_tools,
            name=name,
            created_at=created_at,
            dataset_id=dataset_id,
            trace_span_id=trace_span_id,
        )

        example.additional_properties = d
        return example

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
