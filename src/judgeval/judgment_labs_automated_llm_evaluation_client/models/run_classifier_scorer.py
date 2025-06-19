from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.example import Example
    from ..models.run_classifier_scorer_conversation_item import (
        RunClassifierScorerConversationItem,
    )
    from ..models.run_classifier_scorer_options import RunClassifierScorerOptions


T = TypeVar("T", bound="RunClassifierScorer")


@_attrs_define
class RunClassifierScorer:
    """
    Attributes:
        conversation (list['RunClassifierScorerConversationItem']):
        options (RunClassifierScorerOptions):
        example (Example):
        model (str):
    """

    conversation: list["RunClassifierScorerConversationItem"]
    options: "RunClassifierScorerOptions"
    example: "Example"
    model: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        conversation = []
        for conversation_item_data in self.conversation:
            conversation_item = conversation_item_data.to_dict()
            conversation.append(conversation_item)

        options = self.options.to_dict()

        example = self.example.to_dict()

        model = self.model

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "conversation": conversation,
                "options": options,
                "example": example,
                "model": model,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.example import Example
        from ..models.run_classifier_scorer_conversation_item import (
            RunClassifierScorerConversationItem,
        )
        from ..models.run_classifier_scorer_options import RunClassifierScorerOptions

        d = dict(src_dict)
        conversation = []
        _conversation = d.pop("conversation")
        for conversation_item_data in _conversation:
            conversation_item = RunClassifierScorerConversationItem.from_dict(
                conversation_item_data
            )

            conversation.append(conversation_item)

        options = RunClassifierScorerOptions.from_dict(d.pop("options"))

        example = Example.from_dict(d.pop("example"))

        model = d.pop("model")

        run_classifier_scorer = cls(
            conversation=conversation,
            options=options,
            example=example,
            model=model,
        )

        run_classifier_scorer.additional_properties = d
        return run_classifier_scorer

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
