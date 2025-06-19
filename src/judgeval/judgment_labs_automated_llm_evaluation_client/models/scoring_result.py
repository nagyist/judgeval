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
    from ..models.custom_example import CustomExample
    from ..models.example import Example
    from ..models.scorer_data import ScorerData
    from ..models.trace_span import TraceSpan


T = TypeVar("T", bound="ScoringResult")


@_attrs_define
class ScoringResult:
    """A ScoringResult contains the output of one or more scorers applied to a single example.
    Ie: One input, one actual_output, one expected_output, etc..., and 1+ scorer (Faithfulness, Hallucination,
    Summarization, etc...)

    Args:
        success (bool): Whether the evaluation was successful.
                        This means that all scorers applied to this example returned a success.
        scorer_data (List[ScorerData]): The scorers data for the evaluated example
        data_object (Optional[Example]): The original example object that was used to create the ScoringResult, can be
    Example, CustomExample (future), WorkflowRun (future)


        Attributes:
            success (bool):
            scorers_data (Union[None, list['ScorerData']]):
            name (Union[None, Unset, str]):
            data_object (Union['CustomExample', 'Example', 'TraceSpan', None, Unset]):
            trace_id (Union[None, Unset, str]):
            run_duration (Union[None, Unset, float]):
            evaluation_cost (Union[None, Unset, float]):
    """

    success: bool
    scorers_data: Union[None, list["ScorerData"]]
    name: Union[None, Unset, str] = UNSET
    data_object: Union["CustomExample", "Example", "TraceSpan", None, Unset] = UNSET
    trace_id: Union[None, Unset, str] = UNSET
    run_duration: Union[None, Unset, float] = UNSET
    evaluation_cost: Union[None, Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.custom_example import CustomExample
        from ..models.example import Example
        from ..models.trace_span import TraceSpan

        success = self.success

        scorers_data: Union[None, list[dict[str, Any]]]
        if isinstance(self.scorers_data, list):
            scorers_data = []
            for scorers_data_type_0_item_data in self.scorers_data:
                scorers_data_type_0_item = scorers_data_type_0_item_data.to_dict()
                scorers_data.append(scorers_data_type_0_item)

        else:
            scorers_data = self.scorers_data

        name: Union[None, Unset, str]
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        data_object: Union[None, Unset, dict[str, Any]]
        if isinstance(self.data_object, Unset):
            data_object = UNSET
        elif isinstance(self.data_object, TraceSpan):
            data_object = self.data_object.to_dict()
        elif isinstance(self.data_object, CustomExample):
            data_object = self.data_object.to_dict()
        elif isinstance(self.data_object, Example):
            data_object = self.data_object.to_dict()
        else:
            data_object = self.data_object

        trace_id: Union[None, Unset, str]
        if isinstance(self.trace_id, Unset):
            trace_id = UNSET
        else:
            trace_id = self.trace_id

        run_duration: Union[None, Unset, float]
        if isinstance(self.run_duration, Unset):
            run_duration = UNSET
        else:
            run_duration = self.run_duration

        evaluation_cost: Union[None, Unset, float]
        if isinstance(self.evaluation_cost, Unset):
            evaluation_cost = UNSET
        else:
            evaluation_cost = self.evaluation_cost

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "success": success,
                "scorers_data": scorers_data,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if data_object is not UNSET:
            field_dict["data_object"] = data_object
        if trace_id is not UNSET:
            field_dict["trace_id"] = trace_id
        if run_duration is not UNSET:
            field_dict["run_duration"] = run_duration
        if evaluation_cost is not UNSET:
            field_dict["evaluation_cost"] = evaluation_cost

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.custom_example import CustomExample
        from ..models.example import Example
        from ..models.scorer_data import ScorerData
        from ..models.trace_span import TraceSpan

        d = dict(src_dict)
        success = d.pop("success")

        def _parse_scorers_data(data: object) -> Union[None, list["ScorerData"]]:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                scorers_data_type_0 = []
                _scorers_data_type_0 = data
                for scorers_data_type_0_item_data in _scorers_data_type_0:
                    scorers_data_type_0_item = ScorerData.from_dict(
                        scorers_data_type_0_item_data
                    )

                    scorers_data_type_0.append(scorers_data_type_0_item)

                return scorers_data_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, list["ScorerData"]], data)

        scorers_data = _parse_scorers_data(d.pop("scorers_data"))

        def _parse_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_data_object(
            data: object,
        ) -> Union["CustomExample", "Example", "TraceSpan", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_object_type_0 = TraceSpan.from_dict(data)

                return data_object_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_object_type_1 = CustomExample.from_dict(data)

                return data_object_type_1
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_object_type_2 = Example.from_dict(data)

                return data_object_type_2
            except:  # noqa: E722
                pass
            return cast(
                Union["CustomExample", "Example", "TraceSpan", None, Unset], data
            )

        data_object = _parse_data_object(d.pop("data_object", UNSET))

        def _parse_trace_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        trace_id = _parse_trace_id(d.pop("trace_id", UNSET))

        def _parse_run_duration(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        run_duration = _parse_run_duration(d.pop("run_duration", UNSET))

        def _parse_evaluation_cost(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        evaluation_cost = _parse_evaluation_cost(d.pop("evaluation_cost", UNSET))

        scoring_result = cls(
            success=success,
            scorers_data=scorers_data,
            name=name,
            data_object=data_object,
            trace_id=trace_id,
            run_duration=run_duration,
            evaluation_cost=evaluation_cost,
        )

        scoring_result.additional_properties = d
        return scoring_result

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
