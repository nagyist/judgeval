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
    from ..models.example import Example
    from ..models.trace import Trace


T = TypeVar("T", bound="DatasetPush")


@_attrs_define
class DatasetPush:
    """
    Attributes:
        dataset_alias (str):
        project_name (str):
        comments (Union[None, Unset, str]):
        source_file (Union[None, Unset, str]):
        examples (Union[None, Unset, list['Example']]):
        traces (Union[None, Unset, list['Trace']]):
        is_trace (Union[Unset, bool]):  Default: False.
        overwrite (Union[None, Unset, bool]):  Default: False.
    """

    dataset_alias: str
    project_name: str
    comments: Union[None, Unset, str] = UNSET
    source_file: Union[None, Unset, str] = UNSET
    examples: Union[None, Unset, list["Example"]] = UNSET
    traces: Union[None, Unset, list["Trace"]] = UNSET
    is_trace: Union[Unset, bool] = False
    overwrite: Union[None, Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        dataset_alias = self.dataset_alias

        project_name = self.project_name

        comments: Union[None, Unset, str]
        if isinstance(self.comments, Unset):
            comments = UNSET
        else:
            comments = self.comments

        source_file: Union[None, Unset, str]
        if isinstance(self.source_file, Unset):
            source_file = UNSET
        else:
            source_file = self.source_file

        examples: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.examples, Unset):
            examples = UNSET
        elif isinstance(self.examples, list):
            examples = []
            for examples_type_0_item_data in self.examples:
                examples_type_0_item = examples_type_0_item_data.to_dict()
                examples.append(examples_type_0_item)

        else:
            examples = self.examples

        traces: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.traces, Unset):
            traces = UNSET
        elif isinstance(self.traces, list):
            traces = []
            for traces_type_0_item_data in self.traces:
                traces_type_0_item = traces_type_0_item_data.to_dict()
                traces.append(traces_type_0_item)

        else:
            traces = self.traces

        is_trace = self.is_trace

        overwrite: Union[None, Unset, bool]
        if isinstance(self.overwrite, Unset):
            overwrite = UNSET
        else:
            overwrite = self.overwrite

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "dataset_alias": dataset_alias,
                "project_name": project_name,
            }
        )
        if comments is not UNSET:
            field_dict["comments"] = comments
        if source_file is not UNSET:
            field_dict["source_file"] = source_file
        if examples is not UNSET:
            field_dict["examples"] = examples
        if traces is not UNSET:
            field_dict["traces"] = traces
        if is_trace is not UNSET:
            field_dict["is_trace"] = is_trace
        if overwrite is not UNSET:
            field_dict["overwrite"] = overwrite

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.example import Example
        from ..models.trace import Trace

        d = dict(src_dict)
        dataset_alias = d.pop("dataset_alias")

        project_name = d.pop("project_name")

        def _parse_comments(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        comments = _parse_comments(d.pop("comments", UNSET))

        def _parse_source_file(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        source_file = _parse_source_file(d.pop("source_file", UNSET))

        def _parse_examples(data: object) -> Union[None, Unset, list["Example"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                examples_type_0 = []
                _examples_type_0 = data
                for examples_type_0_item_data in _examples_type_0:
                    examples_type_0_item = Example.from_dict(examples_type_0_item_data)

                    examples_type_0.append(examples_type_0_item)

                return examples_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["Example"]], data)

        examples = _parse_examples(d.pop("examples", UNSET))

        def _parse_traces(data: object) -> Union[None, Unset, list["Trace"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                traces_type_0 = []
                _traces_type_0 = data
                for traces_type_0_item_data in _traces_type_0:
                    traces_type_0_item = Trace.from_dict(traces_type_0_item_data)

                    traces_type_0.append(traces_type_0_item)

                return traces_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["Trace"]], data)

        traces = _parse_traces(d.pop("traces", UNSET))

        is_trace = d.pop("is_trace", UNSET)

        def _parse_overwrite(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        overwrite = _parse_overwrite(d.pop("overwrite", UNSET))

        dataset_push = cls(
            dataset_alias=dataset_alias,
            project_name=project_name,
            comments=comments,
            source_file=source_file,
            examples=examples,
            traces=traces,
            is_trace=is_trace,
            overwrite=overwrite,
        )

        dataset_push.additional_properties = d
        return dataset_push

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
