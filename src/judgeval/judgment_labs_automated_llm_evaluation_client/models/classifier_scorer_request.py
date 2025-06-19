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
    from ..models.classifier_scorer_request_options import (
        ClassifierScorerRequestOptions,
    )
    from ..models.message_item import MessageItem


T = TypeVar("T", bound="ClassifierScorerRequest")


@_attrs_define
class ClassifierScorerRequest:
    """
    Attributes:
        name (str):
        conversation (list['MessageItem']):
        options (ClassifierScorerRequestOptions):
        slug (Union[None, Unset, str]):
    """

    name: str
    conversation: list["MessageItem"]
    options: "ClassifierScorerRequestOptions"
    slug: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        conversation = []
        for conversation_item_data in self.conversation:
            conversation_item = conversation_item_data.to_dict()
            conversation.append(conversation_item)

        options = self.options.to_dict()

        slug: Union[None, Unset, str]
        if isinstance(self.slug, Unset):
            slug = UNSET
        else:
            slug = self.slug

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "conversation": conversation,
                "options": options,
            }
        )
        if slug is not UNSET:
            field_dict["slug"] = slug

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.classifier_scorer_request_options import (
            ClassifierScorerRequestOptions,
        )
        from ..models.message_item import MessageItem

        d = dict(src_dict)
        name = d.pop("name")

        conversation = []
        _conversation = d.pop("conversation")
        for conversation_item_data in _conversation:
            conversation_item = MessageItem.from_dict(conversation_item_data)

            conversation.append(conversation_item)

        options = ClassifierScorerRequestOptions.from_dict(d.pop("options"))

        def _parse_slug(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        slug = _parse_slug(d.pop("slug", UNSET))

        classifier_scorer_request = cls(
            name=name,
            conversation=conversation,
            options=options,
            slug=slug,
        )

        classifier_scorer_request.additional_properties = d
        return classifier_scorer_request

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
