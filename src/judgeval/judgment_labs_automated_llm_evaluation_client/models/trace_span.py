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
    from ..models.tool import Tool
    from ..models.trace_span_additional_metadata_type_0 import (
        TraceSpanAdditionalMetadataType0,
    )
    from ..models.trace_span_annotation_type_0_item import TraceSpanAnnotationType0Item
    from ..models.trace_span_error_type_0 import TraceSpanErrorType0
    from ..models.trace_span_inputs_type_0 import TraceSpanInputsType0
    from ..models.trace_span_state_after_type_0 import TraceSpanStateAfterType0
    from ..models.trace_span_state_before_type_0 import TraceSpanStateBeforeType0
    from ..models.trace_usage import TraceUsage


T = TypeVar("T", bound="TraceSpan")


@_attrs_define
class TraceSpan:
    """
    Attributes:
        span_id (str):
        trace_id (str):
        function (str):
        depth (int):
        created_at (Union[Any, None, Unset]):
        parent_span_id (Union[None, Unset, str]):
        span_type (Union[None, Unset, str]):  Default: 'span'.
        inputs (Union['TraceSpanInputsType0', None, Unset]):
        error (Union['TraceSpanErrorType0', None, Unset]):
        output (Union[Any, None, Unset]):
        usage (Union['TraceUsage', None, Unset]):
        duration (Union[None, Unset, float]):
        annotation (Union[None, Unset, list['TraceSpanAnnotationType0Item']]):
        expected_tools (Union[None, Unset, list['Tool']]):
        additional_metadata (Union['TraceSpanAdditionalMetadataType0', None, Unset]):
        has_evaluation (Union[None, Unset, bool]):  Default: False.
        agent_name (Union[None, Unset, str]):
        state_before (Union['TraceSpanStateBeforeType0', None, Unset]):
        state_after (Union['TraceSpanStateAfterType0', None, Unset]):
    """

    span_id: str
    trace_id: str
    function: str
    depth: int
    created_at: Union[Any, None, Unset] = UNSET
    parent_span_id: Union[None, Unset, str] = UNSET
    span_type: Union[None, Unset, str] = "span"
    inputs: Union["TraceSpanInputsType0", None, Unset] = UNSET
    error: Union["TraceSpanErrorType0", None, Unset] = UNSET
    output: Union[Any, None, Unset] = UNSET
    usage: Union["TraceUsage", None, Unset] = UNSET
    duration: Union[None, Unset, float] = UNSET
    annotation: Union[None, Unset, list["TraceSpanAnnotationType0Item"]] = UNSET
    expected_tools: Union[None, Unset, list["Tool"]] = UNSET
    additional_metadata: Union["TraceSpanAdditionalMetadataType0", None, Unset] = UNSET
    has_evaluation: Union[None, Unset, bool] = False
    agent_name: Union[None, Unset, str] = UNSET
    state_before: Union["TraceSpanStateBeforeType0", None, Unset] = UNSET
    state_after: Union["TraceSpanStateAfterType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.trace_span_additional_metadata_type_0 import (
            TraceSpanAdditionalMetadataType0,
        )
        from ..models.trace_span_error_type_0 import TraceSpanErrorType0
        from ..models.trace_span_inputs_type_0 import TraceSpanInputsType0
        from ..models.trace_span_state_after_type_0 import TraceSpanStateAfterType0
        from ..models.trace_span_state_before_type_0 import TraceSpanStateBeforeType0
        from ..models.trace_usage import TraceUsage

        span_id = self.span_id

        trace_id = self.trace_id

        function = self.function

        depth = self.depth

        created_at: Union[Any, None, Unset]
        if isinstance(self.created_at, Unset):
            created_at = UNSET
        else:
            created_at = self.created_at

        parent_span_id: Union[None, Unset, str]
        if isinstance(self.parent_span_id, Unset):
            parent_span_id = UNSET
        else:
            parent_span_id = self.parent_span_id

        span_type: Union[None, Unset, str]
        if isinstance(self.span_type, Unset):
            span_type = UNSET
        else:
            span_type = self.span_type

        inputs: Union[None, Unset, dict[str, Any]]
        if isinstance(self.inputs, Unset):
            inputs = UNSET
        elif isinstance(self.inputs, TraceSpanInputsType0):
            inputs = self.inputs.to_dict()
        else:
            inputs = self.inputs

        error: Union[None, Unset, dict[str, Any]]
        if isinstance(self.error, Unset):
            error = UNSET
        elif isinstance(self.error, TraceSpanErrorType0):
            error = self.error.to_dict()
        else:
            error = self.error

        output: Union[Any, None, Unset]
        if isinstance(self.output, Unset):
            output = UNSET
        else:
            output = self.output

        usage: Union[None, Unset, dict[str, Any]]
        if isinstance(self.usage, Unset):
            usage = UNSET
        elif isinstance(self.usage, TraceUsage):
            usage = self.usage.to_dict()
        else:
            usage = self.usage

        duration: Union[None, Unset, float]
        if isinstance(self.duration, Unset):
            duration = UNSET
        else:
            duration = self.duration

        annotation: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.annotation, Unset):
            annotation = UNSET
        elif isinstance(self.annotation, list):
            annotation = []
            for annotation_type_0_item_data in self.annotation:
                annotation_type_0_item = annotation_type_0_item_data.to_dict()
                annotation.append(annotation_type_0_item)

        else:
            annotation = self.annotation

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

        additional_metadata: Union[None, Unset, dict[str, Any]]
        if isinstance(self.additional_metadata, Unset):
            additional_metadata = UNSET
        elif isinstance(self.additional_metadata, TraceSpanAdditionalMetadataType0):
            additional_metadata = self.additional_metadata.to_dict()
        else:
            additional_metadata = self.additional_metadata

        has_evaluation: Union[None, Unset, bool]
        if isinstance(self.has_evaluation, Unset):
            has_evaluation = UNSET
        else:
            has_evaluation = self.has_evaluation

        agent_name: Union[None, Unset, str]
        if isinstance(self.agent_name, Unset):
            agent_name = UNSET
        else:
            agent_name = self.agent_name

        state_before: Union[None, Unset, dict[str, Any]]
        if isinstance(self.state_before, Unset):
            state_before = UNSET
        elif isinstance(self.state_before, TraceSpanStateBeforeType0):
            state_before = self.state_before.to_dict()
        else:
            state_before = self.state_before

        state_after: Union[None, Unset, dict[str, Any]]
        if isinstance(self.state_after, Unset):
            state_after = UNSET
        elif isinstance(self.state_after, TraceSpanStateAfterType0):
            state_after = self.state_after.to_dict()
        else:
            state_after = self.state_after

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "function": function,
                "depth": depth,
            }
        )
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if parent_span_id is not UNSET:
            field_dict["parent_span_id"] = parent_span_id
        if span_type is not UNSET:
            field_dict["span_type"] = span_type
        if inputs is not UNSET:
            field_dict["inputs"] = inputs
        if error is not UNSET:
            field_dict["error"] = error
        if output is not UNSET:
            field_dict["output"] = output
        if usage is not UNSET:
            field_dict["usage"] = usage
        if duration is not UNSET:
            field_dict["duration"] = duration
        if annotation is not UNSET:
            field_dict["annotation"] = annotation
        if expected_tools is not UNSET:
            field_dict["expected_tools"] = expected_tools
        if additional_metadata is not UNSET:
            field_dict["additional_metadata"] = additional_metadata
        if has_evaluation is not UNSET:
            field_dict["has_evaluation"] = has_evaluation
        if agent_name is not UNSET:
            field_dict["agent_name"] = agent_name
        if state_before is not UNSET:
            field_dict["state_before"] = state_before
        if state_after is not UNSET:
            field_dict["state_after"] = state_after

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tool import Tool
        from ..models.trace_span_additional_metadata_type_0 import (
            TraceSpanAdditionalMetadataType0,
        )
        from ..models.trace_span_annotation_type_0_item import (
            TraceSpanAnnotationType0Item,
        )
        from ..models.trace_span_error_type_0 import TraceSpanErrorType0
        from ..models.trace_span_inputs_type_0 import TraceSpanInputsType0
        from ..models.trace_span_state_after_type_0 import TraceSpanStateAfterType0
        from ..models.trace_span_state_before_type_0 import TraceSpanStateBeforeType0
        from ..models.trace_usage import TraceUsage

        d = dict(src_dict)
        span_id = d.pop("span_id")

        trace_id = d.pop("trace_id")

        function = d.pop("function")

        depth = d.pop("depth")

        def _parse_created_at(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        created_at = _parse_created_at(d.pop("created_at", UNSET))

        def _parse_parent_span_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        parent_span_id = _parse_parent_span_id(d.pop("parent_span_id", UNSET))

        def _parse_span_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        span_type = _parse_span_type(d.pop("span_type", UNSET))

        def _parse_inputs(data: object) -> Union["TraceSpanInputsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                inputs_type_0 = TraceSpanInputsType0.from_dict(data)

                return inputs_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceSpanInputsType0", None, Unset], data)

        inputs = _parse_inputs(d.pop("inputs", UNSET))

        def _parse_error(data: object) -> Union["TraceSpanErrorType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                error_type_0 = TraceSpanErrorType0.from_dict(data)

                return error_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceSpanErrorType0", None, Unset], data)

        error = _parse_error(d.pop("error", UNSET))

        def _parse_output(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        output = _parse_output(d.pop("output", UNSET))

        def _parse_usage(data: object) -> Union["TraceUsage", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                usage_type_0 = TraceUsage.from_dict(data)

                return usage_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceUsage", None, Unset], data)

        usage = _parse_usage(d.pop("usage", UNSET))

        def _parse_duration(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        duration = _parse_duration(d.pop("duration", UNSET))

        def _parse_annotation(
            data: object,
        ) -> Union[None, Unset, list["TraceSpanAnnotationType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                annotation_type_0 = []
                _annotation_type_0 = data
                for annotation_type_0_item_data in _annotation_type_0:
                    annotation_type_0_item = TraceSpanAnnotationType0Item.from_dict(
                        annotation_type_0_item_data
                    )

                    annotation_type_0.append(annotation_type_0_item)

                return annotation_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["TraceSpanAnnotationType0Item"]], data)

        annotation = _parse_annotation(d.pop("annotation", UNSET))

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

        def _parse_additional_metadata(
            data: object,
        ) -> Union["TraceSpanAdditionalMetadataType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                additional_metadata_type_0 = TraceSpanAdditionalMetadataType0.from_dict(
                    data
                )

                return additional_metadata_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceSpanAdditionalMetadataType0", None, Unset], data)

        additional_metadata = _parse_additional_metadata(
            d.pop("additional_metadata", UNSET)
        )

        def _parse_has_evaluation(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        has_evaluation = _parse_has_evaluation(d.pop("has_evaluation", UNSET))

        def _parse_agent_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        agent_name = _parse_agent_name(d.pop("agent_name", UNSET))

        def _parse_state_before(
            data: object,
        ) -> Union["TraceSpanStateBeforeType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                state_before_type_0 = TraceSpanStateBeforeType0.from_dict(data)

                return state_before_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceSpanStateBeforeType0", None, Unset], data)

        state_before = _parse_state_before(d.pop("state_before", UNSET))

        def _parse_state_after(
            data: object,
        ) -> Union["TraceSpanStateAfterType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                state_after_type_0 = TraceSpanStateAfterType0.from_dict(data)

                return state_after_type_0
            except:  # noqa: E722
                pass
            return cast(Union["TraceSpanStateAfterType0", None, Unset], data)

        state_after = _parse_state_after(d.pop("state_after", UNSET))

        trace_span = cls(
            span_id=span_id,
            trace_id=trace_id,
            function=function,
            depth=depth,
            created_at=created_at,
            parent_span_id=parent_span_id,
            span_type=span_type,
            inputs=inputs,
            error=error,
            output=output,
            usage=usage,
            duration=duration,
            annotation=annotation,
            expected_tools=expected_tools,
            additional_metadata=additional_metadata,
            has_evaluation=has_evaluation,
            agent_name=agent_name,
            state_before=state_before,
            state_after=state_after,
        )

        trace_span.additional_properties = d
        return trace_span

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
