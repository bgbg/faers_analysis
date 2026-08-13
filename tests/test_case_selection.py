"""Tests for lab issue #4: inclusion/exclusion case selection (R23/R24).

The fixture case set and the exclusion-question pairing structure below
mirror the shape of the exclusion sidecar built for methodology entry
31(B2) (``config/exclusions.csv`` in the ``pesher`` repo: a question's
inclusion PT set paired with an exclusion PT set drawn from a sibling,
narrower sheet in the same workbook), but the term strings are synthetic
placeholders -- not MedDRA content -- so this repo carries no licensed
data and no cross-repo runtime dependency on ``pesher``.
"""

import json
import os

import pandas as pd
import pytest

from src.mark_data import mark_data, mark_drug_data, mark_reaction_data
from src.utils import ContingencyMatrix, QuestionConfig

# ---------------------------------------------------------------------------
# QuestionConfig: parsing the exclusion set E
# ---------------------------------------------------------------------------


def test_question_config_json_parses_exclusion(tmp_path):
    fn = tmp_path / "q.json"
    fn.write_text(
        json.dumps(
            {
                "drug": ["Drug A"],
                "reaction": ["Term I ", "Term I 2"],
                "exclusion": ["Term E"],
            }
        )
    )
    config = QuestionConfig.config_from_json_file(str(fn))
    assert config.reactions == ["term i", "term i 2"]
    assert config.exclusions == ["term e"]


def test_question_config_json_without_exclusion_key_is_none(tmp_path):
    fn = tmp_path / "q.json"
    fn.write_text(json.dumps({"drug": ["Drug A"], "reaction": ["Term I"]}))
    config = QuestionConfig.config_from_json_file(str(fn))
    assert config.exclusions is None


def test_question_config_excel_parses_exclusion_column(tmp_path):
    fn = tmp_path / "q.xlsx"
    with pd.ExcelWriter(fn) as writer:
        pd.DataFrame(
            {
                "drug": ["Drug A", None],
                "reaction": ["Term I", None],
                "exclusion": ["Term E", None],
            }
        ).to_excel(writer, sheet_name="only I", index=False)
    configs = QuestionConfig.configs_from_excel_file(str(fn))
    assert len(configs) == 1
    assert configs[0].exclusions == ["term e"]
    assert configs[0].control is None


def test_question_config_excel_control_and_exclusion_together(tmp_path):
    fn = tmp_path / "q.xlsx"
    with pd.ExcelWriter(fn) as writer:
        pd.DataFrame(
            {
                "drug": ["Drug A"],
                "reaction": ["Term I"],
                "control": ["Drug B"],
                "exclusion": ["Term E"],
            }
        ).to_excel(writer, sheet_name="full", index=False)
    configs = QuestionConfig.configs_from_excel_file(str(fn))
    assert configs[0].control == ["drug b"]
    assert configs[0].exclusions == ["term e"]


def test_question_config_excel_no_exclusion_column_backward_compatible(tmp_path):
    fn = tmp_path / "q.xlsx"
    with pd.ExcelWriter(fn) as writer:
        pd.DataFrame({"drug": ["Drug A"], "reaction": ["Term I"]}).to_excel(
            writer, sheet_name="plain", index=False
        )
    configs = QuestionConfig.configs_from_excel_file(str(fn))
    assert configs[0].exclusions is None
    assert configs[0].control is None


# ---------------------------------------------------------------------------
# Case counting: the four I/E membership combinations, both semantics
# ---------------------------------------------------------------------------

DRUG = "drug a"
TERM_I = "term i"
TERM_E = "term e"
TERM_OTHER = "term other"


def _build_marked_data(exclusions):
    """8 cases: {exposed, unexposed} x {I-only, E-only, both, neither}."""
    config = QuestionConfig(
        name="q",
        drugs=[DRUG],
        reactions=[TERM_I],
        control=None,
        exclusions=exclusions,
    )

    # caseid -> (exposed, reaction PTs reported)
    cases = {
        "1_exposed_i_only": (True, [TERM_I]),
        "2_exposed_e_only": (True, [TERM_E]),
        "3_exposed_both": (True, [TERM_I, TERM_E]),
        "4_exposed_neither": (True, [TERM_OTHER]),
        "5_unexposed_i_only": (False, [TERM_I]),
        "6_unexposed_e_only": (False, [TERM_E]),
        "7_unexposed_both": (False, [TERM_I, TERM_E]),
        "8_unexposed_neither": (False, [TERM_OTHER]),
    }

    drug_rows = []
    reac_rows = []
    for caseid, (exposed, pts) in cases.items():
        drug_rows.append({"caseid": caseid, "drugname": DRUG if exposed else "drug z"})
        for pt in pts:
            reac_rows.append({"caseid": caseid, "pt": pt})

    df_drug_raw = pd.DataFrame(drug_rows)
    df_reac_raw = pd.DataFrame(reac_rows)

    drug_names = {DRUG, "drug z"}
    reaction_types = {TERM_I, TERM_E, TERM_OTHER}

    df_drug = mark_drug_data(df_drug_raw, drug_names)
    df_reac = mark_reaction_data(df_reac_raw, reaction_types)

    df_demo = pd.DataFrame({"q": ["2020q1"] * len(cases)}, index=list(cases.keys()))
    df_demo.index.name = "caseid"

    marked = mark_data(
        df_drug=df_drug, df_reac=df_reac, df_demo=df_demo, config_items=[config]
    )
    return marked, config


def test_marked_data_has_excluded_column_only_when_configured():
    marked, config = _build_marked_data(exclusions=[TERM_E])
    assert f"excluded {config.name}" in marked.columns

    marked_no_e, config_no_e = _build_marked_data(exclusions=None)
    assert f"excluded {config_no_e.name}" not in marked_no_e.columns


def test_eligibility_semantics_drops_any_case_reporting_e():
    marked, config = _build_marked_data(exclusions=[TERM_E])
    cm = ContingencyMatrix.from_results_table(
        marked, config, exclusion_semantics="eligibility"
    )
    a = cm.get_count_value(True, True)
    b = cm.get_count_value(True, False)
    c = cm.get_count_value(False, True)
    d = cm.get_count_value(False, False)
    # Only the I-only and neither cases (both arms) survive; E-only and
    # both-I-and-E cases are removed entirely, from both arms.
    assert (a, b, c, d) == (1, 1, 1, 1)
    assert a + b + c + d == 4


def test_outcome_semantics_keeps_all_cases_forces_outcome_negative():
    marked, config = _build_marked_data(exclusions=[TERM_E])
    cm = ContingencyMatrix.from_results_table(
        marked, config, exclusion_semantics="outcome"
    )
    a = cm.get_count_value(True, True)
    b = cm.get_count_value(True, False)
    c = cm.get_count_value(False, True)
    d = cm.get_count_value(False, False)
    # All 8 cases stay. A case reporting E is forced outcome-negative
    # regardless of I, so only the two I-only cases are outcome-positive.
    assert (a, b, c, d) == (1, 3, 1, 3)
    assert a + b + c + d == 8


def test_default_semantics_is_eligibility_beta():
    """Methodology entry 31(D4): (beta) eligibility is primary."""
    marked, config = _build_marked_data(exclusions=[TERM_E])
    cm_default = ContingencyMatrix.from_results_table(marked, config)
    cm_eligibility = ContingencyMatrix.from_results_table(
        marked, config, exclusion_semantics="eligibility"
    )
    pd.testing.assert_frame_equal(cm_default.tbl, cm_eligibility.tbl)


def test_unknown_exclusion_semantics_raises():
    marked, config = _build_marked_data(exclusions=[TERM_E])
    with pytest.raises(ValueError):
        ContingencyMatrix.from_results_table(
            marked, config, exclusion_semantics="bogus"
        )


# ---------------------------------------------------------------------------
# R24: empty E must reproduce the pre-#4 pipeline exactly, on a fixed quarter
# ---------------------------------------------------------------------------


def _pre_change_crosstab(data, config):
    """Verbatim replica of `ContingencyMatrix.from_results_table` as it was
    before lab issue #4 (no exclusion handling at all) -- the golden
    oracle for the R24 backwards-compatibility check.
    """
    exposure = data[f"exposed {config.name}"]
    exposure.name = "exposure"
    outcome = data[f"reacted {config.name}"]
    outcome.name = "outcome"
    crosstab = (
        pd.crosstab(exposure, outcome)
        .reindex([False, True], axis=0, fill_value=0)
        .reindex([False, True], axis=1, fill_value=0)
    )
    return ContingencyMatrix(crosstab)


def test_empty_exclusion_identical_to_pre_change_pipeline_on_fixed_quarter():
    marked, config = _build_marked_data(exclusions=None)

    golden = _pre_change_crosstab(marked, config)

    for semantics in ("eligibility", "outcome"):
        candidate = ContingencyMatrix.from_results_table(
            marked, config, exclusion_semantics=semantics
        )
        pd.testing.assert_frame_equal(candidate.tbl, golden.tbl)
        assert candidate.ror() == golden.ror()
        assert candidate.crosstab().equals(golden.crosstab())
