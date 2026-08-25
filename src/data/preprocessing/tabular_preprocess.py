"""Preprocessing for raw TCGA-BRCA clinical rows into :class:`ClinicalData`.

Raw-source assumption (one per module): the input is a flat mapping of column
name -> value as produced by reading a TCGA clinical table row with pandas
(e.g. ``df.to_dict(orient="records")``). Values may be missing (``None`` /
``NaN`` / empty string), strings, or numbers. Downstream code should feed one
patient's row to :meth:`ClinicalPreprocessor.process`.

The clinical contract is exactly four variables: age, sex, pathological
tumour stage, pathological nodal stage. Stage values are collapsed onto the
five-way vocabularies in :mod:`src.data.schema.clinical`.

Stage collapsing rules
----------------------
Tumour stage: TCGA records substages (``STAGE IIA``, ``STAGE IIIC``); these
collapse to their parent stage (``II``, ``III``). ``STAGE X`` and blanks
become ``UNKNOWN``.

Nodal stage: TCGA records subcategories (``N1A``, ``N1MI``, ``N0 (I+)``,
``N0 (MOL+)``). These collapse to their parent category, which follows AJCC:
``N0(i+)`` and ``N0(mol+)`` denote isolated tumour cells / molecular-only
findings and are staged as N0, whereas ``N1mi`` (micrometastasis) is staged
as N1. ``NX`` and blanks become ``UNKNOWN``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Final

from src.data.schema.clinical import ClinicalData, NodalStage, Sex, TumorStage
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SEX_MALE_TOKENS: Final[frozenset[str]] = frozenset({"m", "male"})
_SEX_FEMALE_TOKENS: Final[frozenset[str]] = frozenset({"f", "female"})

#: Roman parent stage -> enum member. Arabic input is converted first.
_TUMOR_STAGE_BY_ROMAN: Final[Mapping[str, TumorStage]] = {
    "I": TumorStage.STAGE_I,
    "II": TumorStage.STAGE_II,
    "III": TumorStage.STAGE_III,
    "IV": TumorStage.STAGE_IV,
}
_ARABIC_TO_ROMAN: Final[Mapping[str, str]] = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
}

#: Tokens that explicitly encode "stage not determined".
_TUMOR_STAGE_UNKNOWN_TOKENS: Final[frozenset[str]] = frozenset(
    {"X", "STAGEX", "NA", "UNKNOWN", "NOTAVAILABLE", "NOTREPORTED", "DISCREPANCY"}
)
_NODAL_UNKNOWN_TOKENS: Final[frozenset[str]] = frozenset(
    {"NX", "NA", "UNKNOWN", "NOTAVAILABLE", "NOTREPORTED"}
)

_NODAL_STAGE_BY_DIGIT: Final[Mapping[str, NodalStage]] = {
    "0": NodalStage.N0,
    "1": NodalStage.N1,
    "2": NodalStage.N2,
    "3": NodalStage.N3,
}

#: Nodal subcategories whose parent digit does NOT determine the stage.
#: ``N0(i+)`` / ``N0(mol+)`` stay N0 by the digit rule, so only the
#: micrometastasis case needs to be named explicitly; it is listed here for
#: documentation value and to guard against a future digit-rule change.
_NODAL_SUBCATEGORY_OVERRIDES: Final[Mapping[str, NodalStage]] = {
    "N1MI": NodalStage.N1,
}


def _is_missing(value: Any) -> bool:
    """Return ``True`` if ``value`` is ``None``, NaN, or an empty string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


class ClinicalPreprocessor:
    """Cleans raw BRCA clinical rows and constructs validated ``ClinicalData``.

    The preprocessor is deterministic and stateless: the same raw row always
    yields the same record.
    """

    #: Explicit allow-list of raw column names this preprocessor will read.
    #: Nothing outside this set is consulted, so no outcome/label column can
    #: leak into the feature path (CLAUDE.md section 11).
    ALLOWED_FIELDS: Final[frozenset[str]] = frozenset(
        {"age", "sex", "tumor_stage", "nodal_stage"}
    )

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialise the preprocessor.

        Args:
            config: Optional ``configs/breast/data.yaml`` sub-mapping,
                reserved for future overrides; unused today.
        """
        self._config = dict(config) if config is not None else {}

    def process(self, raw: Mapping[str, Any]) -> ClinicalData:
        """Build a validated :class:`ClinicalData` record from a raw row.

        Args:
            raw: Flat mapping containing at least ``age``, ``sex``,
                ``tumor_stage`` and ``nodal_stage``. Extra keys are ignored.

        Returns:
            A validated ``ClinicalData`` instance.

        Raises:
            KeyError: If a required field is missing from ``raw``.
            ValueError: If a value cannot be coerced or fails validation.
        """
        return ClinicalData(
            age=self._require_int(raw, "age"),
            sex=self._normalize_sex(self._require(raw, "sex")),
            tumor_stage=self.normalize_tumor_stage(raw.get("tumor_stage")),
            nodal_stage=self.normalize_nodal_stage(raw.get("nodal_stage")),
        )

    @staticmethod
    def _require(raw: Mapping[str, Any], key: str) -> Any:
        """Return ``raw[key]`` or raise loudly if missing/NaN/blank."""
        if key not in raw or _is_missing(raw[key]):
            raise KeyError(f"Required clinical field '{key}' is missing.")
        return raw[key]

    def _require_int(self, raw: Mapping[str, Any], key: str) -> int:
        """Return a required field coerced to ``int``."""
        value = self._require(raw, key)
        try:
            return int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Clinical field '{key}'={value!r} is not numeric."
            ) from exc

    @staticmethod
    def _normalize_sex(value: Any) -> Sex:
        """Map a raw sex token to the :class:`Sex` enum."""
        token = str(value).strip().lower()
        if token in _SEX_MALE_TOKENS:
            return Sex.MALE
        if token in _SEX_FEMALE_TOKENS:
            return Sex.FEMALE
        raise ValueError(f"Unrecognised sex value: {value!r}")

    @staticmethod
    def normalize_tumor_stage(value: Any) -> TumorStage:
        """Collapse a raw AJCC tumour-stage token to its parent stage.

        Args:
            value: Raw token such as ``"STAGE IIA"``, ``"3"``, ``"STAGE X"``,
                or a missing value.

        Returns:
            The collapsed :class:`TumorStage`. Missing and explicitly
            indeterminate values return :attr:`TumorStage.UNKNOWN`.

        Raises:
            ValueError: If the token is present but not recognisable as a
                stage. Unrecognised values are never silently bucketed into
                ``UNKNOWN`` (CLAUDE.md section 13).
        """
        if _is_missing(value):
            return TumorStage.UNKNOWN

        token = re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())
        if token in _TUMOR_STAGE_UNKNOWN_TOKENS:
            return TumorStage.UNKNOWN

        token = re.sub(r"^STAGE", "", token)
        if token in _TUMOR_STAGE_UNKNOWN_TOKENS or not token:
            return TumorStage.UNKNOWN

        match = re.fullmatch(r"(IV|III|II|I|[1-4])([A-C]?)", token)
        if not match:
            raise ValueError(f"Unrecognised tumour stage: {value!r}")

        base = _ARABIC_TO_ROMAN.get(match.group(1), match.group(1))
        return _TUMOR_STAGE_BY_ROMAN[base]

    @staticmethod
    def normalize_nodal_stage(value: Any) -> NodalStage:
        """Collapse a raw AJCC nodal-stage token to its parent category.

        Args:
            value: Raw token such as ``"N1A"``, ``"N0 (I+)"``, ``"N1MI"``,
                ``"NX"``, or a missing value.

        Returns:
            The collapsed :class:`NodalStage`. Missing and ``NX`` values
            return :attr:`NodalStage.UNKNOWN`.

        Raises:
            ValueError: If the token is present but not recognisable as a
                nodal stage.
        """
        if _is_missing(value):
            return NodalStage.UNKNOWN

        token = re.sub(r"[^A-Z0-9+-]", "", str(value).strip().upper())
        if token in _NODAL_UNKNOWN_TOKENS or not token:
            return NodalStage.UNKNOWN

        override = _NODAL_SUBCATEGORY_OVERRIDES.get(token)
        if override is not None:
            return override

        # N<digit> optionally followed by a subcategory: a letter suffix
        # (N1A), a micrometastasis marker (MI), or an isolated-tumour-cell /
        # molecular marker (I+, I-, MOL+).
        match = re.fullmatch(r"N([0-3])(MI|MOL[+-]|I[+-]|[A-C])?", token)
        if not match:
            raise ValueError(f"Unrecognised nodal stage: {value!r}")

        return _NODAL_STAGE_BY_DIGIT[match.group(1)]
