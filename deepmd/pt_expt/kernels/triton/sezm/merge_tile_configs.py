# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201
r"""Merge shape-sweep output into the built-in launch-configuration table.

:mod:`.sweep_tile_configs` prints its winners as a ``tile_config_data``
fragment::

    # tile_config_data.py fragment (merge under the GPU name):
        "K500SM_AI": {
            "gate": {
                (32, 1): (32, 4, 2),
            },
        },

Pasting that by hand is where a tuning campaign goes wrong: a dropped line, a
fragment merged under the wrong device name, or a family renamed on one side
all produce a table that resolves silently to *something* and is then trusted.
This module performs the merge mechanically and refuses to write a table it
cannot read back.

The edit is deliberately conservative. The target file is parsed with
:mod:`ast` and every change is a single-row edit: an existing ``key: value``
line is rewritten in place, a new key is inserted at its sorted position, and
a family the device did not have is appended as a block. Comments, the
hand-written family order and the ``# fmt: off`` guard survive, and the
reviewable diff is exactly the swept rows. After writing, the new device
subtree must read back to precisely the parsed fragment and every other device
entry must compare equal to before.

Usage
-----
::

    # dry run: print the diff and the consistency report
    python -m deepmd.pt_expt.kernels.triton.sezm.merge_tile_configs \\
        --from-log /tmp/sweep_l2.log --from-log /tmp/sweep_l3.log

    # apply, forcing the device key the table will be looked up by
    python -m deepmd.pt_expt.kernels.triton.sezm.merge_tile_configs \\
        --from-log /tmp/sweep_l2.log --device K500SM_AI --write

Fragments already present for the device are merged key by key, so a level-2
sweep and a later level-3 sweep can be folded in one after another; keys the
fragment does not mention are kept. ``--replace`` renders the whole device
entry instead, which is what a full regeneration wants.
"""

from __future__ import (
    annotations,
)

import argparse
import ast
import difflib
import json
from pathlib import (
    Path,
)
import sys
from typing import (
    Any,
)

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _MODULE_DIR / "tile_config_data.py"
_DEFAULT_CONFIGS = _MODULE_DIR / "tile_configs.py"
_TABLE_NAME = "BUILTIN_TILE_CONFIGS"
_FAMILIES_NAME = "TILE_CONFIG_FAMILIES"
_FRAGMENT_MARKER = "# tile_config_data.py fragment"

_DEVICE_INDENT = "    "
_FAMILY_INDENT = "        "
_ROW_INDENT = "            "


class MergeError(ValueError):
    """A fragment or the target table cannot be merged safely."""


# ----------------------------------------------------------------------
# Fragment parsing
# ----------------------------------------------------------------------
def _fragment_body(text: str) -> str:
    """Return the fragment lines of a sweep log or a saved fragment file."""
    lines = text.splitlines()
    markers = [index for index, line in enumerate(lines) if _FRAGMENT_MARKER in line]
    if markers:
        lines = lines[markers[-1] + 1 :]
    body: list[str] = []
    for line in lines:
        if not line.strip():
            body.append(line)
            continue
        # A fragment is entirely indented; the first flush-left line ends it.
        if not line.startswith((" ", "\t")):
            break
        body.append(line)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        raise MergeError(
            "no tile_config_data fragment found; pass the sweep stdout via "
            "--from-log or a saved fragment file via --fragment"
        )
    return "\n".join(body)


def parse_fragment(text: str, *, source: str = "<text>") -> dict[str, Any]:
    """Parse one or more ``tile_config_data`` fragments into a mapping.

    Parameters
    ----------
    text : str
        Sweep stdout (with the fragment marker) or a saved fragment file.
    source : str
        Name used in error messages.

    Returns
    -------
    dict
        ``{device_name: {family: {shape_key: config_or_None}}}``.
    """
    body = _fragment_body(text)
    try:
        parsed = ast.literal_eval("{\n" + body + "\n}")
    except (SyntaxError, ValueError) as error:
        raise MergeError(
            f"{source}: fragment is not a literal mapping: {error}"
        ) from error
    if not isinstance(parsed, dict) or not parsed:
        raise MergeError(f"{source}: fragment holds no device entries")
    return parsed


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def known_families(configs_path: Path = _DEFAULT_CONFIGS) -> set[str]:
    """Read ``TILE_CONFIG_FAMILIES`` from the query module without importing it.

    Importing :mod:`.tile_configs` pulls in Torch; the family list is a plain
    tuple literal, so the merge tool reads it statically and needs neither a
    device nor the kernel stack.
    """
    try:
        tree = ast.parse(configs_path.read_text())
    except OSError as error:
        raise MergeError(f"cannot read {configs_path}: {error}") from error
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id == _FAMILIES_NAME:
                return set(ast.literal_eval(value))
    raise MergeError(f"{configs_path}: {_FAMILIES_NAME} not found")


def _is_int_tuple(entry: object) -> bool:
    """Return whether ``entry`` is a tuple of integers, nested as allowed.

    ``stack_fp32`` stores one launch tuple per stage slot, so a value there is a
    tuple of tuples; every other family stores a flat tuple.  Both are accepted,
    a list or any non-integer leaf is not.
    """
    if isinstance(entry, int) and not isinstance(entry, bool):
        return True
    return (
        isinstance(entry, tuple)
        and bool(entry)
        and all(_is_int_tuple(item) for item in entry)
    )


def _config_arity(entry: object) -> int | None:
    return len(entry) if isinstance(entry, tuple) else None


def validate_entries(
    mapping: dict[str, Any],
    *,
    families: set[str],
    existing: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Check a parsed fragment against the table's own conventions.

    Returns ``(errors, warnings)``.  Errors block the write; warnings describe
    what cannot be decided mechanically (the arity of a family that appears for
    the first time in this table).
    """
    errors: list[str] = []
    warnings: list[str] = []
    existing = existing or {}
    arities: dict[str, set[int]] = {}
    for devices in existing.values():
        for family, table in devices.items():
            for entry in table.values():
                arity = _config_arity(entry)
                if arity is not None:
                    arities.setdefault(family, set()).add(arity)

    for device, devices in mapping.items():
        if not isinstance(device, str) or not device.strip():
            errors.append(f"device key {device!r} is not a nonempty string")
            continue
        if not isinstance(devices, dict) or not devices:
            errors.append(f"{device}: entry is not a nonempty family mapping")
            continue
        for family, table in devices.items():
            if not isinstance(family, str):
                errors.append(f"{device}: family key {family!r} is not a string")
                continue
            if family not in families:
                errors.append(
                    f"{device}/{family}: unknown family; expected one of "
                    f"{sorted(families)}"
                )
            if not isinstance(table, dict) or not table:
                errors.append(f"{device}/{family}: entry is not a nonempty key mapping")
                continue
            for key, entry in table.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not all(isinstance(item, int) for item in key)
                ):
                    errors.append(
                        f"{device}/{family}: key {key!r} is not a 2-integer tuple"
                    )
                    continue
                if entry is not None and not _is_int_tuple(entry):
                    errors.append(
                        f"{device}/{family}/{key}: config {entry!r} is neither None nor "
                        "a tuple of integers"
                    )
                    continue
                arity = _config_arity(entry)
                if arity is None:
                    continue
                known = arities.get(family)
                if known is None:
                    warnings.append(
                        f"{device}/{family}: new family for this table; "
                        f"per-config arity {arity} cannot be cross-checked"
                    )
                elif arity not in known:
                    errors.append(
                        f"{device}/{family}/{key}: config arity {arity} disagrees with "
                        f"the table's {sorted(known)} for this family"
                    )
    return errors, warnings


# ----------------------------------------------------------------------
# Source navigation
# ----------------------------------------------------------------------
def _find_table(tree: ast.Module) -> ast.Dict:
    """Return the ``BUILTIN_TILE_CONFIGS`` dict literal of a parsed module."""
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id == _TABLE_NAME:
                if not isinstance(value, ast.Dict):
                    raise MergeError(f"{_TABLE_NAME} is not a dict literal")
                return value
    raise MergeError(f"{_TABLE_NAME} not found")


def _find_device(outer: ast.Dict, device: str) -> ast.Dict | None:
    """Return the value node of ``device``'s entry, or ``None``."""
    for key, value in zip(outer.keys, outer.values):
        if isinstance(key, ast.Constant) and key.value == device:
            if not isinstance(value, ast.Dict):
                raise MergeError(f"device entry {device!r} is not a dict literal")
            return value
    return None


def _find_family(device_dict: ast.Dict, family: str) -> ast.Dict | None:
    """Return the value node of one family table inside a device entry."""
    for key, value in zip(device_dict.keys, device_dict.values):
        if isinstance(key, ast.Constant) and key.value == family:
            if not isinstance(value, ast.Dict):
                raise MergeError(f"family entry {family!r} is not a dict literal")
            return value
    return None


def _table_rows(table: ast.Dict) -> list[tuple[tuple[int, int], int, int]]:
    """Return ``(shape_key, start_line, stop_line)`` per row of a family table."""
    rows = []
    for key, value in zip(table.keys, table.values):
        if isinstance(key, ast.Tuple):
            rows.append((ast.literal_eval(key), value.lineno - 1, value.end_lineno))
    return rows


def _current_row(
    source: str, device: str, family: str, key: tuple[int, int]
) -> tuple[bool, Any]:
    """Return ``(present, value)`` for one row of the current source."""
    outer = _find_table(ast.parse(source))
    device_dict = _find_device(outer, device)
    if device_dict is None:
        return False, None
    table = _find_family(device_dict, family)
    if table is None:
        return False, None
    for key_node, value_node in zip(table.keys, table.values):
        if isinstance(key_node, ast.Tuple) and ast.literal_eval(key_node) == key:
            return True, ast.literal_eval(value_node)
    return False, None


# ----------------------------------------------------------------------
# Source editing
# ----------------------------------------------------------------------
def render_family(family: str, table: dict[Any, Any]) -> list[str]:
    """Render one family table in the table's canonical layout."""
    lines = [f'{_FAMILY_INDENT}"{family}": {{']
    for key in sorted(table):
        lines.append(f"{_ROW_INDENT}{key!r}: {table[key]!r},")
    lines.append(f"{_FAMILY_INDENT}}},")
    return lines


def render_device(device: str, families: dict[str, Any]) -> list[str]:
    """Render one device entry in the table's canonical layout."""
    lines = [f'{_DEVICE_INDENT}"{device}": {{']
    for family in sorted(families):
        lines.extend(render_family(family, families[family]))
    lines.append(f"{_DEVICE_INDENT}}},")
    return lines


def _apply_row(
    source: str, device: str, family: str, key: tuple[int, int], entry: Any
) -> str:
    """Write one ``key: entry`` row, as an in-place update or an insertion."""
    lines = source.splitlines(keepends=True)
    outer = _find_table(ast.parse(source))

    device_dict = _find_device(outer, device)
    if device_dict is None:
        insert_at = outer.end_lineno - 1
        lines[insert_at:insert_at] = [
            line + "\n" for line in render_device(device, {family: {key: entry}})
        ]
        return "".join(lines)

    table = _find_family(device_dict, family)
    if table is None:
        insert_at = device_dict.end_lineno - 1
        lines[insert_at:insert_at] = [
            line + "\n" for line in render_family(family, {key: entry})
        ]
        return "".join(lines)

    rows = _table_rows(table)
    for existing, start, stop in rows:
        if existing != key:
            continue
        indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        lines[start:stop] = [f"{indent}{key!r}: {entry!r},\n"]
        return "".join(lines)

    insert_at = table.end_lineno - 1
    for existing, start, _stop in rows:
        if key < existing:
            insert_at = start
            break
    lines[insert_at:insert_at] = [f"{_ROW_INDENT}{key!r}: {entry!r},\n"]
    return "".join(lines)


def _merge_device(
    source: str, device: str, families: dict[str, Any], changes: dict[str, Any]
) -> str:
    """Fold one device's fragment into ``source``, one row at a time."""
    for family in sorted(families):
        for key in sorted(families[family]):
            entry = families[family][key]
            present, current = _current_row(source, device, family, key)
            if not present:
                changes[device]["added"][f"{family}{key}"] = entry
            elif current != entry:
                changes[device]["updated"][f"{family}{key}"] = entry
            source = _apply_row(source, device, family, key, entry)
    return source


def _replace_device(source: str, device: str, families: dict[str, Any]) -> str:
    """Replace an existing device entry with a fully rendered block."""
    lines = source.splitlines(keepends=True)
    outer = _find_table(ast.parse(source))
    rendered = [line + "\n" for line in render_device(device, families)]
    device_dict = _find_device(outer, device)
    if device_dict is None:
        insert_at = outer.end_lineno - 1
    else:
        insert_at = device_dict.lineno - 1
        del lines[insert_at : device_dict.end_lineno]
    lines[insert_at:insert_at] = rendered
    return "".join(lines)


def merge_source(
    source: str,
    mapping: dict[str, Any],
    *,
    replace: bool = False,
    path: str = "tile_config_data.py",
) -> tuple[str, dict[str, Any]]:
    """Return ``(new_source, changes)`` for merging ``mapping`` into ``source``.

    Parameters
    ----------
    source : str
        Current text of ``tile_config_data.py``.
    mapping : dict
        Parsed fragment, as returned by :func:`parse_fragment`.
    replace : bool
        Replace an existing device entry instead of merging key by key.
    path : str
        Name used in error messages.

    Returns
    -------
    tuple
        The new file text and a per-device report of added/updated keys.
    """
    before = ast.literal_eval(_find_table(ast.parse(source)))
    new_source = source
    changes: dict[str, Any] = {}
    for device, families in mapping.items():
        changes[device] = {"added": {}, "updated": {}, "created": device not in before}
        if replace:
            new_source = _replace_device(new_source, device, families)
            for family, table in families.items():
                for key, entry in table.items():
                    changes[device]["added"][f"{family}{key}"] = entry
        else:
            new_source = _merge_device(new_source, device, families, changes)
    _verify(new_source, mapping, before, replace=replace, path=path)
    return new_source, changes


def _verify(
    new_source: str,
    mapping: dict[str, Any],
    before: dict[str, Any],
    *,
    replace: bool,
    path: str,
) -> None:
    """Refuse a result that does not read back to what was requested."""
    try:
        after = ast.literal_eval(_find_table(ast.parse(new_source)))
    except (SyntaxError, ValueError) as error:
        raise MergeError(f"{path}: merged table no longer parses: {error}") from error
    for device, families in mapping.items():
        stored = after.get(device)
        if not isinstance(stored, dict):
            raise MergeError(f"{path}: device {device!r} missing from the merged table")
        for family, table in families.items():
            for key, entry in table.items():
                if stored.get(family, {}).get(key, "absent") != entry:
                    raise MergeError(
                        f"{path}: {device}/{family}/{key} reads back as "
                        f"{stored.get(family, {}).get(key, 'absent')!r}, "
                        f"expected {entry!r}"
                    )
        if replace and set(stored) != set(families):
            raise MergeError(
                f"{path}: {device!r} kept families "
                f"{sorted(set(stored) - set(families))} that --replace should have dropped"
            )
    for device, families in before.items():
        if device in mapping:
            continue
        if after.get(device) != families:
            raise MergeError(f"{path}: untouched device {device!r} changed")
    if len(after) < len(before):
        raise MergeError(f"{path}: merged table lost device entries")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _load_fragments(logs: list[Path], fragments: list[Path]) -> dict[str, Any]:
    """Parse and fold every input into one device mapping."""
    mapping: dict[str, Any] = {}
    inputs = [(log, "log") for log in logs] + [(path, "fragment") for path in fragments]
    if not inputs:
        raise MergeError("pass at least one --from-log or --fragment")
    for path, kind in inputs:
        try:
            text = path.read_text()
        except OSError as error:
            raise MergeError(f"cannot read {kind} {path}: {error}") from error
        parsed = parse_fragment(text, source=str(path))
        for device, families in parsed.items():
            target = mapping.setdefault(device, {})
            for family, table in families.items():
                target.setdefault(family, {}).update(table)
    return mapping


def _report(changes: dict[str, Any]) -> str:
    """Render the per-device merge report."""
    lines = []
    for device, info in sorted(changes.items()):
        verb = "created" if info["created"] else "merged"
        lines.append(
            f"device {device!r}: {verb}, "
            f"{len(info['added'])} added, {len(info['updated'])} updated"
        )
        for key in sorted(info["updated"]):
            lines.append(f"  updated {key} -> {info['updated'][key]!r}")
        if not info["created"] and not info["added"] and not info["updated"]:
            lines.append("  no change")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Merge fragments into the table; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-log",
        type=Path,
        action="append",
        default=[],
        help="sweep stdout; the tile_config_data fragment is extracted",
    )
    parser.add_argument(
        "--fragment",
        type=Path,
        action="append",
        default=[],
        help="saved fragment file (no marker line needed)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="rename the fragment's device key, e.g. K500SM_AI",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help="tile_config_data.py to merge into",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=_DEFAULT_CONFIGS,
        help="tile_configs.py holding the family list",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing device entry instead of merging key by row",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the file; without it the merge is a dry run",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    try:
        mapping = _load_fragments(args.from_log, args.fragment)
        if args.device is not None:
            if len(mapping) != 1:
                raise MergeError(
                    f"--device renames one fragment, but {len(mapping)} devices were "
                    f"found: {sorted(mapping)}"
                )
            mapping = {args.device: next(iter(mapping.values()))}
        source = args.data.read_text()
        families = known_families(args.configs)
        current = ast.literal_eval(_find_table(ast.parse(source)))
        errors, warnings = validate_entries(
            mapping, families=families, existing=current
        )
    except MergeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print("\nfragment rejected; nothing written", file=sys.stderr)
        return 1

    try:
        new_source, changes = merge_source(
            source, mapping, replace=args.replace, path=str(args.data)
        )
    except MergeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(_report(changes))
    if args.write:
        if new_source != source:
            args.data.write_text(new_source)
            print(f"written: {args.data}")
        else:
            print("no change; file left untouched")
    else:
        sys.stdout.writelines(
            difflib.unified_diff(
                source.splitlines(keepends=True),
                new_source.splitlines(keepends=True),
                fromfile=str(args.data),
                tofile=str(args.data) + " (merged)",
            )
        )
        print("\ndry run: pass --write to apply")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "data": str(args.data),
                    "write": bool(args.write),
                    "changes": changes,
                    "warnings": warnings,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
