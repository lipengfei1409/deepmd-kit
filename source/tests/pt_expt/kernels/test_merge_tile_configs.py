# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host-side tests for the tile-configuration merge tool.

The merge tool edits a shipped data file, so what is verified here is the
edit itself: the fragment parser, the conventions the table is validated
against, and -- for a temporary copy of the real table -- that the merged text
reads back to exactly the requested entries while every other device entry
stays byte-identical.  No device, no Torch and no Triton are involved.
"""

from __future__ import (
    annotations,
)

import ast
import importlib.util
from pathlib import (
    Path,
)

import pytest

# The tool is loaded by path: importing the ``sezm`` package would pull in the
# Torch/Triton kernel modules, and this test is about text editing.
_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "deepmd"
    / "pt_expt"
    / "kernels"
    / "triton"
    / "sezm"
    / "merge_tile_configs.py"
)
_SPEC = importlib.util.spec_from_file_location("merge_tile_configs", _MODULE_PATH)
merge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge)

_TILE_DATA = _MODULE_PATH.with_name("tile_config_data.py")

_LOG = """\
sweep device: K500SM_AI | wave_size=64
  BM= 16 warps= 2 stages= 2:    0.512 ms  <-
BEST gate[(32, 1)] = (16, 2, 2)  # 0.512 ms

# tile_config_data.py fragment (merge under the GPU name):
    "K500SM_AI": {
        "gate": {
            (32, 1): (16, 2, 2),
            (32, 2): None,
        },
        "stack_fp32": {
            (64, 2): ((64, 64, 32, 4, 2), (64, 64, 32, 4, 2), (64, 64, 32, 4, 2)),
        },
    },

"""


def test_fragment_is_extracted_from_a_sweep_log() -> None:
    parsed = merge.parse_fragment(_LOG, source="log")
    assert set(parsed) == {"K500SM_AI"}
    families = parsed["K500SM_AI"]
    assert families["gate"] == {(32, 1): (16, 2, 2), (32, 2): None}
    assert families["stack_fp32"][(64, 2)][0] == (64, 64, 32, 4, 2)


def test_fragment_file_without_marker_is_accepted() -> None:
    body = _LOG.split("# tile_config_data.py fragment", 1)[1].split("\n", 1)[1]
    assert merge.parse_fragment(body, source="fragment") == merge.parse_fragment(_LOG)


def test_missing_fragment_is_reported() -> None:
    with pytest.raises(merge.MergeError, match="no tile_config_data fragment"):
        merge.parse_fragment("sweep finished\n", source="log")


def test_non_literal_fragment_is_rejected() -> None:
    with pytest.raises(merge.MergeError, match="not a literal mapping"):
        merge.parse_fragment('    "K500SM_AI": {"gate": build()},\n', source="log")


def test_unknown_family_is_an_error() -> None:
    mapping = {"K500SM_AI": {"not_a_family": {(32, 1): (16, 2, 2)}}}
    errors, _ = merge.validate_entries(mapping, families={"gate"})
    assert any("unknown family" in error for error in errors)


def test_arity_mismatch_against_the_existing_table_is_an_error() -> None:
    existing = {"NVIDIA H20": {"gate": {(32, 1): (32, 4, 2)}}}
    mapping = {"K500SM_AI": {"gate": {(32, 1): (16, 2)}}}
    errors, _ = merge.validate_entries(mapping, families={"gate"}, existing=existing)
    assert any("arity" in error for error in errors)


def test_new_family_is_a_warning_not_an_error() -> None:
    mapping = {"K500SM_AI": {"gate": {(32, 1): (16, 2, 2, 1)}}}
    errors, warnings = merge.validate_entries(
        mapping, families={"gate"}, existing={"NVIDIA H20": {"stack_fp32": {}}}
    )
    assert not errors
    assert any("new family" in warning for warning in warnings)


def test_bad_key_and_bad_config_are_errors() -> None:
    mapping = {"K500SM_AI": {"gate": {"32": (16, 2, 2), (32, 1, 9): (16, 2, 2)}}}
    errors, _ = merge.validate_entries(mapping, families={"gate"})
    assert any("2-integer tuple" in error for error in errors)
    mapping = {"K500SM_AI": {"gate": {(32, 1): [16, 2, 2]}}}
    errors, _ = merge.validate_entries(mapping, families={"gate"})
    assert any("tuple of integers" in error for error in errors)


def _current_table(source: str) -> dict:
    return ast.literal_eval(merge._find_table(ast.parse(source)))


def test_merge_creates_a_device_and_leaves_others_untouched() -> None:
    source = _TILE_DATA.read_text()
    before = _current_table(source)
    mapping = merge.parse_fragment(_LOG)
    merged, changes = merge.merge_source(source, mapping)

    after = _current_table(merged)
    assert set(after) == set(before) | {"K500SM_AI"}
    for device, families in before.items():
        assert after[device] == families
    assert after["K500SM_AI"]["gate"] == {(32, 1): (16, 2, 2), (32, 2): None}
    assert changes["K500SM_AI"]["created"] is True
    assert changes["K500SM_AI"]["added"]
    # The file is still a valid module and keeps its formatting guard.
    compile(merged, str(_TILE_DATA), "exec")
    assert "# fmt: off" in merged


def test_merge_into_an_existing_device_updates_only_the_new_keys() -> None:
    mapping = merge.parse_fragment(_LOG)
    source, _ = merge.merge_source(_TILE_DATA.read_text(), mapping)
    later = {"K500SM_AI": {"gate": {(32, 1): (32, 4, 2), (96, 6): (8, 8, 1)}}}
    merged, changes = merge.merge_source(source, later)

    table = _current_table(merged)["K500SM_AI"]
    assert table["gate"][(32, 1)] == (32, 4, 2)
    assert table["gate"][(32, 2)] is None
    assert table["gate"][(96, 6)] == (8, 8, 1)
    assert set(table["stack_fp32"]) == {(64, 2)}
    assert changes["K500SM_AI"]["created"] is False
    assert "gate(32, 1)" in changes["K500SM_AI"]["updated"]
    assert "gate(96, 6)" in changes["K500SM_AI"]["added"]


def _row_order(source: str, device: str, family: str) -> list[tuple[int, int]]:
    outer = merge._find_table(ast.parse(source))
    device_dict = merge._find_device(outer, device)
    assert device_dict is not None
    table = merge._find_family(device_dict, family)
    assert table is not None
    return [ast.literal_eval(key) for key in table.keys]


def test_inserted_rows_keep_the_table_sorted() -> None:
    mapping = merge.parse_fragment(_LOG)
    source, _ = merge.merge_source(_TILE_DATA.read_text(), mapping)
    later = {"K500SM_AI": {"gate": {(16, 1): (8, 1, 1), (96, 6): (8, 8, 1)}}}
    merged, _ = merge.merge_source(source, later)
    order = _row_order(merged, "K500SM_AI", "gate")
    assert order == sorted(order)
    assert set(order) == {(16, 1), (32, 1), (32, 2), (96, 6)}


def test_new_family_keeps_the_existing_device_comments() -> None:
    source = _TILE_DATA.read_text()
    mapping = {"NVIDIA H20": {"stack_m0_gate": {(32, 1): (16, 32, 2, 1)}}}
    merged, changes = merge.merge_source(source, mapping)
    assert changes["NVIDIA H20"]["created"] is False
    assert _current_table(merged)["NVIDIA H20"]["stack_m0_gate"] == {
        (32, 1): (16, 32, 2, 1)
    }
    # The hand-written comment inside the device entry survives the merge.
    assert "# (Cf, lmax) -> (BLOCK_M, num_warps, num_stages)" in merged


def test_merge_is_idempotent() -> None:
    mapping = merge.parse_fragment(_LOG)
    once, _ = merge.merge_source(_TILE_DATA.read_text(), mapping)
    twice, changes = merge.merge_source(once, mapping)
    assert twice == once
    assert not changes["K500SM_AI"]["added"]
    assert not changes["K500SM_AI"]["updated"]


def test_replace_drops_the_previous_entry() -> None:
    mapping = merge.parse_fragment(_LOG)
    source, _ = merge.merge_source(_TILE_DATA.read_text(), mapping)
    replacement = {"K500SM_AI": {"gate": {(32, 1): (8, 1, 1)}}}
    merged, _ = merge.merge_source(source, replacement, replace=True)
    table = _current_table(merged)["K500SM_AI"]
    assert table == {"gate": {(32, 1): (8, 1, 1)}}


def test_multiple_fragments_fold_into_one_device(tmp_path: Path) -> None:
    log_two = tmp_path / "sweep_l3.log"
    log_two.write_text(
        "# tile_config_data.py fragment (merge under the GPU name):\n"
        '    "K500SM_AI": {\n'
        '        "stack_fp16x3": {\n'
        "            (64, 2): None,\n"
        "        },\n"
        "    },\n"
    )
    log_one = tmp_path / "sweep_l2.log"
    log_one.write_text(_LOG)
    mapping = merge._load_fragments([log_one, log_two], [])
    assert set(mapping["K500SM_AI"]) == {"gate", "stack_fp32", "stack_fp16x3"}


def test_cli_dry_run_does_not_write_and_write_applies(tmp_path: Path) -> None:
    data = tmp_path / "tile_config_data.py"
    data.write_text(_TILE_DATA.read_text())
    log = tmp_path / "sweep.log"
    log.write_text(_LOG)
    original = data.read_text()

    assert (
        merge.main(
            ["--from-log", str(log), "--data", str(data), "--device", "K500SM_AI"]
        )
        == 0
    )
    assert data.read_text() == original

    assert (
        merge.main(
            [
                "--from-log",
                str(log),
                "--data",
                str(data),
                "--device",
                "K500SM_AI",
                "--write",
            ]
        )
        == 0
    )
    assert "K500SM_AI" in _current_table(data.read_text())


def test_cli_rejects_a_fragment_that_violates_the_table_conventions(
    tmp_path: Path,
) -> None:
    data = tmp_path / "tile_config_data.py"
    data.write_text(_TILE_DATA.read_text())
    log = tmp_path / "sweep.log"
    log.write_text(
        "# tile_config_data.py fragment (merge under the GPU name):\n"
        '    "K500SM_AI": {\n'
        '        "gate": {\n'
        "            (32, 1): (16, 2),\n"
        "        },\n"
        "    },\n"
    )
    before = data.read_text()
    assert merge.main(["--from-log", str(log), "--data", str(data), "--write"]) == 1
    assert data.read_text() == before
