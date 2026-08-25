"""Contract tests for the canonical PAM50 vocabulary.

:mod:`src.data.pam50` fixes two things that silently invalidate trained
checkpoints if they drift: the *width and order* of the genomic input tensor,
and the *class indices* of the 5-class subtype head. These tests pin both,
plus the two published-symbol aliases the panel needs because Parker et al.
predates the HGNC renames.
"""

from __future__ import annotations

import pytest

from src.data import pam50
from src.data.pam50 import (
    PAM50_GENE_COUNT,
    PAM50_GENE_SET,
    PAM50_GENES,
    PAM50_SUBTYPE_TO_INDEX,
    PAM50_SUBTYPES,
    PUBLISHED_SYMBOL_ALIASES,
    TCGA_SUBTYPE_TOKENS,
    gene_index,
    resolve_gene_symbol,
    subtype_to_index,
    tcga_token_to_index,
)
from src.data.tasks import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    RECEPTOR_TASKS,
    SURVIVAL_TASK,
    TASK_CLASS_LABELS,
    TASK_LOGIT_KEYS,
    TASK_NUM_CLASSES,
)


# --------------------------------------------------------------------------- #
# Gene panel
# --------------------------------------------------------------------------- #
def test_panel_is_exactly_fifty_unique_genes() -> None:
    assert PAM50_GENE_COUNT == 50
    assert len(PAM50_GENES) == 50
    assert len(set(PAM50_GENES)) == 50
    assert PAM50_GENE_SET == frozenset(PAM50_GENES)


def test_gene_order_is_the_documented_sorted_canonical_order() -> None:
    """The order is load-bearing: it indexes the transformer's gene table."""
    assert PAM50_GENES == tuple(sorted(PAM50_GENES))


def test_gene_index_is_dense_and_agrees_with_gene_order() -> None:
    index = gene_index()

    assert sorted(index.values()) == list(range(PAM50_GENE_COUNT))
    assert all(PAM50_GENES[position] == gene for gene, position in index.items())


def test_module_invariants_are_checked_at_import_time() -> None:
    """A broken panel must fail loudly on import, not at training time."""
    pam50._validate_module_invariants()


@pytest.mark.parametrize(
    ("published", "canonical"),
    [("KNTC2", "NDC80"), ("CDCA1", "NUF2")],
)
def test_published_symbol_aliases_resolve(published: str, canonical: str) -> None:
    """Parker et al. predates these two HGNC renames."""
    assert PUBLISHED_SYMBOL_ALIASES[published] == canonical
    assert resolve_gene_symbol(published) == canonical
    assert canonical in PAM50_GENE_SET


def test_alias_keys_are_deprecated_symbols_not_panel_members() -> None:
    assert not set(PUBLISHED_SYMBOL_ALIASES) & PAM50_GENE_SET
    assert set(PUBLISHED_SYMBOL_ALIASES.values()) <= PAM50_GENE_SET


@pytest.mark.parametrize("gene", ["ESR1", "ERBB2", "PGR", "MKI67"])
def test_panel_genes_resolve_to_themselves(gene: str) -> None:
    assert resolve_gene_symbol(gene) == gene


@pytest.mark.parametrize("symbol", ["BRCA1", "TP53", "", "   ", "NDC80X"])
def test_non_panel_symbols_resolve_to_none(symbol: str) -> None:
    """A non-panel row must be skipped, never coerced into the panel."""
    assert resolve_gene_symbol(symbol) is None


def test_resolve_gene_symbol_strips_surrounding_whitespace() -> None:
    assert resolve_gene_symbol("  ESR1 ") == "ESR1"
    assert resolve_gene_symbol(" KNTC2\t") == "NDC80"


# --------------------------------------------------------------------------- #
# Subtype label space
# --------------------------------------------------------------------------- #
def test_subtype_classes_are_the_five_ratified_names_in_order() -> None:
    assert PAM50_SUBTYPES == (
        "Luminal A",
        "Luminal B",
        "HER2-enriched",
        "Basal-like",
        "Normal-like",
    )


def test_subtype_indices_are_dense_and_derived_from_the_class_order() -> None:
    assert PAM50_SUBTYPE_TO_INDEX == {
        "Luminal A": 0,
        "Luminal B": 1,
        "HER2-enriched": 2,
        "Basal-like": 3,
        "Normal-like": 4,
    }


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("BRCA_LumA", 0),
        ("BRCA_LumB", 1),
        ("BRCA_Her2", 2),
        ("BRCA_Basal", 3),
        ("BRCA_Normal", 4),
    ],
)
def test_tcga_subtype_tokens_map_to_class_indices(token: str, expected: int) -> None:
    assert tcga_token_to_index(token) == expected
    assert TCGA_SUBTYPE_TOKENS[token] == PAM50_SUBTYPES[expected]


@pytest.mark.parametrize("token", ["", "  ", "NA", "BRCA_LumC", "Luminal A"])
def test_unrecognised_subtype_token_is_masked_not_defaulted(token: str) -> None:
    """An unmapped token must produce ``None`` (a mask), never class 0."""
    assert tcga_token_to_index(token) is None


def test_subtype_to_index_raises_for_an_unknown_class_name() -> None:
    with pytest.raises(KeyError):
        subtype_to_index("Luminal C")


# --------------------------------------------------------------------------- #
# Task vocabulary
# --------------------------------------------------------------------------- #
def test_task_vocabulary_is_the_five_active_tasks() -> None:
    assert CLASSIFICATION_TASKS == ("subtype", "er", "pr", "her2")
    assert RECEPTOR_TASKS == ("er", "pr", "her2")
    assert SURVIVAL_TASK == "survival"
    assert ALL_TASKS == ("subtype", "er", "pr", "her2", "survival")


def test_task_class_counts_follow_the_pam50_and_receptor_contracts() -> None:
    assert TASK_NUM_CLASSES["subtype"] == len(PAM50_SUBTYPES) == 5
    assert all(TASK_NUM_CLASSES[task] == 2 for task in RECEPTOR_TASKS)


def test_task_class_labels_are_aligned_to_the_class_indices() -> None:
    assert TASK_CLASS_LABELS["subtype"] == PAM50_SUBTYPES
    for task in RECEPTOR_TASKS:
        assert TASK_CLASS_LABELS[task] == ("Negative", "Positive")


def test_logit_keys_match_the_model_output_names() -> None:
    assert TASK_LOGIT_KEYS == {
        "subtype": "subtype_logits",
        "er": "er_logits",
        "pr": "pr_logits",
        "her2": "her2_logits",
    }
