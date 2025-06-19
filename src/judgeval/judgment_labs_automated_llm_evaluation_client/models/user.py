import datetime
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
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.factor import Factor
    from ..models.user_app_metadata import UserAppMetadata
    from ..models.user_identity import UserIdentity
    from ..models.user_user_metadata import UserUserMetadata


T = TypeVar("T", bound="User")


@_attrs_define
class User:
    """
    Attributes:
        id (str):
        app_metadata (UserAppMetadata):
        user_metadata (UserUserMetadata):
        aud (str):
        created_at (datetime.datetime):
        confirmation_sent_at (Union[None, Unset, datetime.datetime]):
        recovery_sent_at (Union[None, Unset, datetime.datetime]):
        email_change_sent_at (Union[None, Unset, datetime.datetime]):
        new_email (Union[None, Unset, str]):
        new_phone (Union[None, Unset, str]):
        invited_at (Union[None, Unset, datetime.datetime]):
        action_link (Union[None, Unset, str]):
        email (Union[None, Unset, str]):
        phone (Union[None, Unset, str]):
        confirmed_at (Union[None, Unset, datetime.datetime]):
        email_confirmed_at (Union[None, Unset, datetime.datetime]):
        phone_confirmed_at (Union[None, Unset, datetime.datetime]):
        last_sign_in_at (Union[None, Unset, datetime.datetime]):
        role (Union[None, Unset, str]):
        updated_at (Union[None, Unset, datetime.datetime]):
        identities (Union[None, Unset, list['UserIdentity']]):
        is_anonymous (Union[Unset, bool]):  Default: False.
        factors (Union[None, Unset, list['Factor']]):
    """

    id: str
    app_metadata: "UserAppMetadata"
    user_metadata: "UserUserMetadata"
    aud: str
    created_at: datetime.datetime
    confirmation_sent_at: Union[None, Unset, datetime.datetime] = UNSET
    recovery_sent_at: Union[None, Unset, datetime.datetime] = UNSET
    email_change_sent_at: Union[None, Unset, datetime.datetime] = UNSET
    new_email: Union[None, Unset, str] = UNSET
    new_phone: Union[None, Unset, str] = UNSET
    invited_at: Union[None, Unset, datetime.datetime] = UNSET
    action_link: Union[None, Unset, str] = UNSET
    email: Union[None, Unset, str] = UNSET
    phone: Union[None, Unset, str] = UNSET
    confirmed_at: Union[None, Unset, datetime.datetime] = UNSET
    email_confirmed_at: Union[None, Unset, datetime.datetime] = UNSET
    phone_confirmed_at: Union[None, Unset, datetime.datetime] = UNSET
    last_sign_in_at: Union[None, Unset, datetime.datetime] = UNSET
    role: Union[None, Unset, str] = UNSET
    updated_at: Union[None, Unset, datetime.datetime] = UNSET
    identities: Union[None, Unset, list["UserIdentity"]] = UNSET
    is_anonymous: Union[Unset, bool] = False
    factors: Union[None, Unset, list["Factor"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        app_metadata = self.app_metadata.to_dict()

        user_metadata = self.user_metadata.to_dict()

        aud = self.aud

        created_at = self.created_at.isoformat()

        confirmation_sent_at: Union[None, Unset, str]
        if isinstance(self.confirmation_sent_at, Unset):
            confirmation_sent_at = UNSET
        elif isinstance(self.confirmation_sent_at, datetime.datetime):
            confirmation_sent_at = self.confirmation_sent_at.isoformat()
        else:
            confirmation_sent_at = self.confirmation_sent_at

        recovery_sent_at: Union[None, Unset, str]
        if isinstance(self.recovery_sent_at, Unset):
            recovery_sent_at = UNSET
        elif isinstance(self.recovery_sent_at, datetime.datetime):
            recovery_sent_at = self.recovery_sent_at.isoformat()
        else:
            recovery_sent_at = self.recovery_sent_at

        email_change_sent_at: Union[None, Unset, str]
        if isinstance(self.email_change_sent_at, Unset):
            email_change_sent_at = UNSET
        elif isinstance(self.email_change_sent_at, datetime.datetime):
            email_change_sent_at = self.email_change_sent_at.isoformat()
        else:
            email_change_sent_at = self.email_change_sent_at

        new_email: Union[None, Unset, str]
        if isinstance(self.new_email, Unset):
            new_email = UNSET
        else:
            new_email = self.new_email

        new_phone: Union[None, Unset, str]
        if isinstance(self.new_phone, Unset):
            new_phone = UNSET
        else:
            new_phone = self.new_phone

        invited_at: Union[None, Unset, str]
        if isinstance(self.invited_at, Unset):
            invited_at = UNSET
        elif isinstance(self.invited_at, datetime.datetime):
            invited_at = self.invited_at.isoformat()
        else:
            invited_at = self.invited_at

        action_link: Union[None, Unset, str]
        if isinstance(self.action_link, Unset):
            action_link = UNSET
        else:
            action_link = self.action_link

        email: Union[None, Unset, str]
        if isinstance(self.email, Unset):
            email = UNSET
        else:
            email = self.email

        phone: Union[None, Unset, str]
        if isinstance(self.phone, Unset):
            phone = UNSET
        else:
            phone = self.phone

        confirmed_at: Union[None, Unset, str]
        if isinstance(self.confirmed_at, Unset):
            confirmed_at = UNSET
        elif isinstance(self.confirmed_at, datetime.datetime):
            confirmed_at = self.confirmed_at.isoformat()
        else:
            confirmed_at = self.confirmed_at

        email_confirmed_at: Union[None, Unset, str]
        if isinstance(self.email_confirmed_at, Unset):
            email_confirmed_at = UNSET
        elif isinstance(self.email_confirmed_at, datetime.datetime):
            email_confirmed_at = self.email_confirmed_at.isoformat()
        else:
            email_confirmed_at = self.email_confirmed_at

        phone_confirmed_at: Union[None, Unset, str]
        if isinstance(self.phone_confirmed_at, Unset):
            phone_confirmed_at = UNSET
        elif isinstance(self.phone_confirmed_at, datetime.datetime):
            phone_confirmed_at = self.phone_confirmed_at.isoformat()
        else:
            phone_confirmed_at = self.phone_confirmed_at

        last_sign_in_at: Union[None, Unset, str]
        if isinstance(self.last_sign_in_at, Unset):
            last_sign_in_at = UNSET
        elif isinstance(self.last_sign_in_at, datetime.datetime):
            last_sign_in_at = self.last_sign_in_at.isoformat()
        else:
            last_sign_in_at = self.last_sign_in_at

        role: Union[None, Unset, str]
        if isinstance(self.role, Unset):
            role = UNSET
        else:
            role = self.role

        updated_at: Union[None, Unset, str]
        if isinstance(self.updated_at, Unset):
            updated_at = UNSET
        elif isinstance(self.updated_at, datetime.datetime):
            updated_at = self.updated_at.isoformat()
        else:
            updated_at = self.updated_at

        identities: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.identities, Unset):
            identities = UNSET
        elif isinstance(self.identities, list):
            identities = []
            for identities_type_0_item_data in self.identities:
                identities_type_0_item = identities_type_0_item_data.to_dict()
                identities.append(identities_type_0_item)

        else:
            identities = self.identities

        is_anonymous = self.is_anonymous

        factors: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.factors, Unset):
            factors = UNSET
        elif isinstance(self.factors, list):
            factors = []
            for factors_type_0_item_data in self.factors:
                factors_type_0_item = factors_type_0_item_data.to_dict()
                factors.append(factors_type_0_item)

        else:
            factors = self.factors

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "app_metadata": app_metadata,
                "user_metadata": user_metadata,
                "aud": aud,
                "created_at": created_at,
            }
        )
        if confirmation_sent_at is not UNSET:
            field_dict["confirmation_sent_at"] = confirmation_sent_at
        if recovery_sent_at is not UNSET:
            field_dict["recovery_sent_at"] = recovery_sent_at
        if email_change_sent_at is not UNSET:
            field_dict["email_change_sent_at"] = email_change_sent_at
        if new_email is not UNSET:
            field_dict["new_email"] = new_email
        if new_phone is not UNSET:
            field_dict["new_phone"] = new_phone
        if invited_at is not UNSET:
            field_dict["invited_at"] = invited_at
        if action_link is not UNSET:
            field_dict["action_link"] = action_link
        if email is not UNSET:
            field_dict["email"] = email
        if phone is not UNSET:
            field_dict["phone"] = phone
        if confirmed_at is not UNSET:
            field_dict["confirmed_at"] = confirmed_at
        if email_confirmed_at is not UNSET:
            field_dict["email_confirmed_at"] = email_confirmed_at
        if phone_confirmed_at is not UNSET:
            field_dict["phone_confirmed_at"] = phone_confirmed_at
        if last_sign_in_at is not UNSET:
            field_dict["last_sign_in_at"] = last_sign_in_at
        if role is not UNSET:
            field_dict["role"] = role
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at
        if identities is not UNSET:
            field_dict["identities"] = identities
        if is_anonymous is not UNSET:
            field_dict["is_anonymous"] = is_anonymous
        if factors is not UNSET:
            field_dict["factors"] = factors

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.factor import Factor
        from ..models.user_app_metadata import UserAppMetadata
        from ..models.user_identity import UserIdentity
        from ..models.user_user_metadata import UserUserMetadata

        d = dict(src_dict)
        id = d.pop("id")

        app_metadata = UserAppMetadata.from_dict(d.pop("app_metadata"))

        user_metadata = UserUserMetadata.from_dict(d.pop("user_metadata"))

        aud = d.pop("aud")

        created_at = isoparse(d.pop("created_at"))

        def _parse_confirmation_sent_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                confirmation_sent_at_type_0 = isoparse(data)

                return confirmation_sent_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        confirmation_sent_at = _parse_confirmation_sent_at(
            d.pop("confirmation_sent_at", UNSET)
        )

        def _parse_recovery_sent_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                recovery_sent_at_type_0 = isoparse(data)

                return recovery_sent_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        recovery_sent_at = _parse_recovery_sent_at(d.pop("recovery_sent_at", UNSET))

        def _parse_email_change_sent_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                email_change_sent_at_type_0 = isoparse(data)

                return email_change_sent_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        email_change_sent_at = _parse_email_change_sent_at(
            d.pop("email_change_sent_at", UNSET)
        )

        def _parse_new_email(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        new_email = _parse_new_email(d.pop("new_email", UNSET))

        def _parse_new_phone(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        new_phone = _parse_new_phone(d.pop("new_phone", UNSET))

        def _parse_invited_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                invited_at_type_0 = isoparse(data)

                return invited_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        invited_at = _parse_invited_at(d.pop("invited_at", UNSET))

        def _parse_action_link(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        action_link = _parse_action_link(d.pop("action_link", UNSET))

        def _parse_email(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        email = _parse_email(d.pop("email", UNSET))

        def _parse_phone(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        phone = _parse_phone(d.pop("phone", UNSET))

        def _parse_confirmed_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                confirmed_at_type_0 = isoparse(data)

                return confirmed_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        confirmed_at = _parse_confirmed_at(d.pop("confirmed_at", UNSET))

        def _parse_email_confirmed_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                email_confirmed_at_type_0 = isoparse(data)

                return email_confirmed_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        email_confirmed_at = _parse_email_confirmed_at(
            d.pop("email_confirmed_at", UNSET)
        )

        def _parse_phone_confirmed_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                phone_confirmed_at_type_0 = isoparse(data)

                return phone_confirmed_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        phone_confirmed_at = _parse_phone_confirmed_at(
            d.pop("phone_confirmed_at", UNSET)
        )

        def _parse_last_sign_in_at(
            data: object,
        ) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_sign_in_at_type_0 = isoparse(data)

                return last_sign_in_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        last_sign_in_at = _parse_last_sign_in_at(d.pop("last_sign_in_at", UNSET))

        def _parse_role(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        role = _parse_role(d.pop("role", UNSET))

        def _parse_updated_at(data: object) -> Union[None, Unset, datetime.datetime]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                updated_at_type_0 = isoparse(data)

                return updated_at_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, datetime.datetime], data)

        updated_at = _parse_updated_at(d.pop("updated_at", UNSET))

        def _parse_identities(data: object) -> Union[None, Unset, list["UserIdentity"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                identities_type_0 = []
                _identities_type_0 = data
                for identities_type_0_item_data in _identities_type_0:
                    identities_type_0_item = UserIdentity.from_dict(
                        identities_type_0_item_data
                    )

                    identities_type_0.append(identities_type_0_item)

                return identities_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["UserIdentity"]], data)

        identities = _parse_identities(d.pop("identities", UNSET))

        is_anonymous = d.pop("is_anonymous", UNSET)

        def _parse_factors(data: object) -> Union[None, Unset, list["Factor"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                factors_type_0 = []
                _factors_type_0 = data
                for factors_type_0_item_data in _factors_type_0:
                    factors_type_0_item = Factor.from_dict(factors_type_0_item_data)

                    factors_type_0.append(factors_type_0_item)

                return factors_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["Factor"]], data)

        factors = _parse_factors(d.pop("factors", UNSET))

        user = cls(
            id=id,
            app_metadata=app_metadata,
            user_metadata=user_metadata,
            aud=aud,
            created_at=created_at,
            confirmation_sent_at=confirmation_sent_at,
            recovery_sent_at=recovery_sent_at,
            email_change_sent_at=email_change_sent_at,
            new_email=new_email,
            new_phone=new_phone,
            invited_at=invited_at,
            action_link=action_link,
            email=email,
            phone=phone,
            confirmed_at=confirmed_at,
            email_confirmed_at=email_confirmed_at,
            phone_confirmed_at=phone_confirmed_at,
            last_sign_in_at=last_sign_in_at,
            role=role,
            updated_at=updated_at,
            identities=identities,
            is_anonymous=is_anonymous,
            factors=factors,
        )

        user.additional_properties = d
        return user

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
