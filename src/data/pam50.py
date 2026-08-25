"""Canonical PAM50 gene vocabulary and molecular-subtype label space.

This module is the single source of truth for the genomic feature contract of
the active TCGA-BRCA pipeline. Two things here are load-bearing and must not be
changed casually:

1.  ``PAM50_GENES`` defines the *width* (50) and the *order* of the genomic
    input tensor. :class:`~src.models.genomics.genomic_transformer.GenomicTransformer`
    holds an ``nn.Embedding(input_dim, d_model)`` gene-identity table whose
    i-th row corresponds to the i-th entry of this list. Reordering the list
    silently invalidates every trained checkpoint.

2.  ``PAM50_SUBTYPES`` defines the integer class indices of the 5-class PAM50
    head. Reordering it silently relabels every prediction.

Provenance
----------
The 50 gene symbols are the Parker et al. classifier set (Parker MJ et al.,
"Supervised risk predictor of breast cancer based on intrinsic subtypes",
J Clin Oncol 2009;27:1160-1167), transcribed from the centroid heatmap in
Appendix Figure A2 of PMC2667820. They are stored here in sorted order, which
is the ordering ratified in the project data contract (CLAUDE.md section 4).
Sorted order is used rather than the publication's display order because it is
reproducible from the symbol set alone and therefore self-documenting.

All 50 symbols resolve directly and uniquely against the ``Hugo_Symbol``
column of ``data_mrna_seq_v2_rsem.txt`` in the TCGA-BRCA PanCancer Atlas
release used by this project. ``PUBLISHED_SYMBOL_ALIASES`` exists because the
publication predates two HGNC renames; it is applied defensively by
:func:`resolve_gene_symbol` so an expression matrix that still uses the
deprecated symbols can be consumed without editing this module.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

#: Number of genes in the PAM50 panel. The genomic input tensor is this wide.
PAM50_GENE_COUNT: Final[int] = 50

#: Canonical, deterministic PAM50 gene ordering. Index i of the genomic
#: feature vector is always ``PAM50_GENES[i]``. Do not reorder.
PAM50_GENES: Final[tuple[str, ...]] = (
    "ACTR3B",
    "ANLN",
    "BAG1",
    "BCL2",
    "BIRC5",
    "BLVRA",
    "CCNB1",
    "CCNE1",
    "CDC20",
    "CDC6",
    "CDH3",
    "CENPF",
    "CEP55",
    "CXXC5",
    "EGFR",
    "ERBB2",
    "ESR1",
    "EXO1",
    "FGFR4",
    "FOXA1",
    "FOXC1",
    "GPR160",
    "GRB7",
    "KIF2C",
    "KRT14",
    "KRT17",
    "KRT5",
    "MAPT",
    "MDM2",
    "MELK",
    "MIA",
    "MKI67",
    "MLPH",
    "MMP11",
    "MYBL2",
    "MYC",
    "NAT1",
    "NDC80",
    "NUF2",
    "ORC6L",
    "PGR",
    "PHGDH",
    "PTTG1",
    "RRM2",
    "SFRP1",
    "SLC39A6",
    "TMEM45B",
    "TYMS",
    "UBE2C",
    "UBE2T",
)

#: Set form of :data:`PAM50_GENES` for O(1) membership tests.
PAM50_GENE_SET: Final[frozenset[str]] = frozenset(PAM50_GENES)

#: Deprecated published symbol -> canonical symbol used in ``PAM50_GENES``.
#: Parker 2009 predates these HGNC renames.
PUBLISHED_SYMBOL_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "KNTC2": "NDC80",
        "CDCA1": "NUF2",
    }
)

#: PAM50 molecular subtype classes in canonical class-index order. Index i is
#: the integer target for the 5-class subtype head.
PAM50_SUBTYPES: Final[tuple[str, ...]] = (
    "Luminal A",
    "Luminal B",
    "HER2-enriched",
    "Basal-like",
    "Normal-like",
)

#: Subtype name -> class index, derived from :data:`PAM50_SUBTYPES`.
PAM50_SUBTYPE_TO_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(PAM50_SUBTYPES)}
)

#: ``SUBTYPE`` tokens as they appear in the PanCancer Atlas
#: ``data_clinical_patient.txt`` -> canonical subtype name.
TCGA_SUBTYPE_TOKENS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "BRCA_LumA": "Luminal A",
        "BRCA_LumB": "Luminal B",
        "BRCA_Her2": "HER2-enriched",
        "BRCA_Basal": "Basal-like",
        "BRCA_Normal": "Normal-like",
    }
)


def _validate_module_invariants() -> None:
    """Fail at import time if the gene/subtype contract is internally broken."""
    if len(PAM50_GENES) != PAM50_GENE_COUNT:
        raise ValueError(
            f"PAM50_GENES has {len(PAM50_GENES)} entries, "
            f"expected {PAM50_GENE_COUNT}."
        )
    if len(set(PAM50_GENES)) != PAM50_GENE_COUNT:
        raise ValueError("PAM50_GENES contains duplicate symbols.")
    if tuple(sorted(PAM50_GENES)) != PAM50_GENES:
        raise ValueError(
            "PAM50_GENES is not in the documented sorted canonical order."
        )
    collisions = set(PUBLISHED_SYMBOL_ALIASES) & set(PAM50_GENES)
    if collisions:
        raise ValueError(
            f"Alias keys must be deprecated symbols, not canonical ones: {collisions}."
        )
    unknown_targets = set(PUBLISHED_SYMBOL_ALIASES.values()) - set(PAM50_GENES)
    if unknown_targets:
        raise ValueError(
            f"Alias targets are not PAM50 genes: {sorted(unknown_targets)}."
        )
    if len(set(PAM50_SUBTYPES)) != len(PAM50_SUBTYPES):
        raise ValueError("PAM50_SUBTYPES contains duplicate class names.")
    unknown_subtypes = set(TCGA_SUBTYPE_TOKENS.values()) - set(PAM50_SUBTYPES)
    if unknown_subtypes:
        raise ValueError(
            f"TCGA subtype tokens map to unknown classes: {sorted(unknown_subtypes)}."
        )


_validate_module_invariants()


def gene_index() -> dict[str, int]:
    """Return a mapping of canonical gene symbol to its feature-vector index.

    Returns:
        Dict mapping each of the 50 canonical symbols to its position in
        :data:`PAM50_GENES`.
    """
    return {symbol: index for index, symbol in enumerate(PAM50_GENES)}


def resolve_gene_symbol(symbol: str) -> str | None:
    """Resolve a raw expression-matrix gene symbol to its canonical PAM50 form.

    Applies :data:`PUBLISHED_SYMBOL_ALIASES` so a matrix using the deprecated
    published symbols (``KNTC2``, ``CDCA1``) still resolves.

    Args:
        symbol: Gene symbol as it appears in the source expression matrix.

    Returns:
        The canonical PAM50 symbol, or ``None`` when ``symbol`` is not part of
        the panel.
    """
    token = symbol.strip()
    if not token:
        return None

    canonical = PUBLISHED_SYMBOL_ALIASES.get(token, token)
    return canonical if canonical in PAM50_GENE_SET else None


def subtype_to_index(subtype: str) -> int:
    """Map a canonical PAM50 subtype name to its integer class index.

    Args:
        subtype: One of :data:`PAM50_SUBTYPES`.

    Returns:
        The class index in ``[0, 5)``.

    Raises:
        KeyError: If ``subtype`` is not a recognised PAM50 class name.
    """
    if subtype not in PAM50_SUBTYPE_TO_INDEX:
        raise KeyError(
            f"Unknown PAM50 subtype {subtype!r}. "
            f"Expected one of {list(PAM50_SUBTYPES)}."
        )
    return PAM50_SUBTYPE_TO_INDEX[subtype]


def tcga_token_to_index(token: str) -> int | None:
    """Map a raw TCGA ``SUBTYPE`` token to a PAM50 class index.

    Args:
        token: Raw token from ``data_clinical_patient.txt`` (e.g.
            ``"BRCA_LumA"``). Whitespace is stripped.

    Returns:
        The class index, or ``None`` when the token is blank or not a
        recognised BRCA subtype call (which must be handled as a masked
        label, never as a default class).
    """
    canonical = TCGA_SUBTYPE_TOKENS.get(token.strip())
    if canonical is None:
        return None
    return PAM50_SUBTYPE_TO_INDEX[canonical]
