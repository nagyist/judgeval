from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.judgment_eval import JudgmentEval
    from ..models.scoring_result import ScoringResult
    from ..models.trace_run import TraceRun


T = TypeVar("T", bound="EvalResults")


@_attrs_define
class EvalResults:
    """
    Attributes:
        results (list['ScoringResult']):
        run (Union['JudgmentEval', 'TraceRun']):
    """

    results: list["ScoringResult"]
    run: Union["JudgmentEval", "TraceRun"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.trace_run import TraceRun

        results = []
        for results_item_data in self.results:
            results_item = results_item_data.to_dict()
            results.append(results_item)

        run: dict[str, Any]
        if isinstance(self.run, TraceRun):
            run = self.run.to_dict()
        else:
            run = self.run.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "results": results,
                "run": run,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.judgment_eval import JudgmentEval
        from ..models.scoring_result import ScoringResult
        from ..models.trace_run import TraceRun

        d = dict(src_dict)
        results = []
        _results = d.pop("results")
        for results_item_data in _results:
            results_item = ScoringResult.from_dict(results_item_data)

            results.append(results_item)

        def _parse_run(data: object) -> Union["JudgmentEval", "TraceRun"]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                run_type_0 = TraceRun.from_dict(data)

                return run_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            run_type_1 = JudgmentEval.from_dict(data)

            return run_type_1

        run = _parse_run(d.pop("run"))

        eval_results = cls(
            results=results,
            run=run,
        )

        eval_results.additional_properties = d
        return eval_results

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
