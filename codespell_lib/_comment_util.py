"""Comment extraction utilities for comments-only spell checking."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

CFAMILY_PROFILE = "c-family"

BUILTIN_EXTENSIONS: dict[str, str] = {
    ".c": CFAMILY_PROFILE,
    ".h": CFAMILY_PROFILE,
    ".cc": CFAMILY_PROFILE,
    ".cpp": CFAMILY_PROFILE,
    ".cxx": CFAMILY_PROFILE,
    ".hh": CFAMILY_PROFILE,
    ".hpp": CFAMILY_PROFILE,
    ".hxx": CFAMILY_PROFILE,
}


class CommentConfigError(Exception):
    """Invalid comment pattern configuration."""


@dataclass(frozen=True)
class MarkerProfile:
    line_markers: tuple[str, ...]
    block_markers: tuple[tuple[str, str], ...]


@dataclass
class CommentRegistry:
    extensions: dict[str, str]
    marker_profiles: dict[str, MarkerProfile]

    def lookup_profile(self, filename: str) -> str | None:
        ext = os.path.splitext(filename)[1].lower()
        return self.extensions.get(ext)

    def mask_text(self, text: str, profile_name: str) -> str:
        if profile_name == CFAMILY_PROFILE:
            return c_family_mask(text)
        profile = self.marker_profiles.get(profile_name)
        if profile is None:
            msg = f"unknown comment profile: {profile_name}"
            raise CommentConfigError(msg)
        return marker_mask(text, profile)


def builtin_registry() -> CommentRegistry:
    return CommentRegistry(
        extensions=dict(BUILTIN_EXTENSIONS),
        marker_profiles={},
    )


def merge_registries(
    base: CommentRegistry, override: CommentRegistry
) -> CommentRegistry:
    extensions = dict(base.extensions)
    extensions.update(override.extensions)
    marker_profiles = dict(base.marker_profiles)
    marker_profiles.update(override.marker_profiles)
    return CommentRegistry(extensions=extensions, marker_profiles=marker_profiles)


def concat_fragments(
    fragments: list[tuple[bool, int, list[str]]],
) -> str:
    return "".join(line for _, _, lines in fragments for line in lines)


def repartition_mask(
    fragments: list[tuple[bool, int, list[str]]],
    full_mask: str,
) -> list[tuple[bool, int, list[str]]]:
    pos = 0
    result: list[tuple[bool, int, list[str]]] = []
    for ignore, line_number, lines in fragments:
        masked_lines: list[str] = []
        for line in lines:
            end = pos + len(line)
            masked_lines.append(full_mask[pos:end])
            pos = end
        result.append((ignore, line_number, masked_lines))
    if pos != len(full_mask):
        msg = "mask length does not match concatenated fragment text"
        raise ValueError(msg)
    return result


def _try_spliced_line_comment(text: str, i: int) -> int | None:
    n = len(text)
    if text[i] != "/":
        return None
    j = i + 1
    if j < n and text[j] == "/":
        return None
    if j >= n or text[j] != "\\":
        return None
    splice = _consume_line_splice(text, j)
    if splice is None or splice >= n or text[splice] != "/":
        return None
    return splice + 1


def _consume_line_splice(text: str, i: int) -> int | None:
    if text[i] != "\\":
        return None
    j = i + 1
    if j >= len(text):
        return None
    if text[j] == "\r":
        if j + 1 < len(text) and text[j + 1] == "\n":
            return j + 2
        return j + 1
    if text[j] == "\n":
        return j + 1
    return None


def _try_raw_string(text: str, i: int) -> tuple[int, str] | None:
    n = len(text)
    j = i
    if j < n and text[j] in "uUL":
        j += 1
        if j < n and text[j] in "uU" and text[j - 1] in "uU":
            j += 1
    if j + 1 >= n or text[j] != "R" or text[j + 1] != '"':
        return None
    j += 2
    delim_start = j
    while j < n and text[j] != "(":
        if text[j] in " \t\r\n()":
            return None
        if j - delim_start > 16:
            return None
        j += 1
    if j >= n:
        return None
    delimiter = text[delim_start:j]
    return j + 1, delimiter


def _raw_string_closes(text: str, i: int, delimiter: str) -> int | None:
    if text[i] != ")":
        return None
    delim_end = i + 1 + len(delimiter)
    if delim_end >= len(text) or text[delim_end] != '"':
        return None
    if text[i + 1 : i + 1 + len(delimiter)] != delimiter:
        return None
    return delim_end + 1


def c_family_mask(text: str) -> str:
    """Return a same-length mask preserving comment text and newlines."""
    result: list[str] = []
    i = 0
    n = len(text)
    state = "normal"
    raw_delimiter = ""

    while i < n:
        splice = _consume_line_splice(text, i)
        if splice is not None:
            if state in ("line_comment", "block_comment"):
                result.append("\\")
                j = i + 1
                if j < n and text[j] == "\r":
                    result.append("\r")
                    j += 1
                if j < n and text[j] == "\n":
                    result.append("\n")
            else:
                result.append(" ")
                j = i + 1
                if j < n and text[j] == "\r":
                    result.append(" ")
                    j += 1
                if j < n and text[j] == "\n":
                    result.append(" ")
            i = splice
            continue

        c = text[i]

        if state == "normal":
            raw_match = _try_raw_string(text, i)
            if raw_match is not None:
                open_end, raw_delimiter = raw_match
                for k in range(i, open_end):
                    result.append(" ")
                state = "raw_string"
                i = open_end
                continue

            if c == '"':
                state = "string"
                result.append(" ")
                i += 1
                continue
            if c == "'":
                state = "char"
                result.append(" ")
                i += 1
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                state = "line_comment"
                result.append("/")
                result.append("/")
                i += 2
                continue
            spliced_comment_end = _try_spliced_line_comment(text, i)
            if spliced_comment_end is not None:
                for k in range(i, spliced_comment_end):
                    result.append(text[k])
                state = "line_comment"
                i = spliced_comment_end
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                state = "block_comment"
                result.append("/")
                result.append("*")
                i += 2
                continue

            result.append(c if c in "\r\n" else " ")
            i += 1
            continue

        if state == "line_comment":
            if c in "\r\n":
                state = "normal"
                result.append(c)
                if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                    result.append("\n")
                    i += 2
                    continue
                i += 1
                continue
            result.append(c)
            i += 1
            continue

        if state == "block_comment":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                result.append("*")
                result.append("/")
                i += 2
                state = "normal"
                continue
            result.append(c if c in "\r\n" else c)
            i += 1
            continue

        if state == "string":
            if c == "\\" and i + 1 < n:
                result.append(" ")
                result.append(" ")
                i += 2
                continue
            if c == '"':
                state = "normal"
                result.append(" ")
                i += 1
                continue
            result.append(c if c in "\r\n" else " ")
            i += 1
            continue

        if state == "char":
            if c == "\\" and i + 1 < n:
                result.append(" ")
                result.append(" ")
                i += 2
                continue
            if c == "'":
                state = "normal"
                result.append(" ")
                i += 1
                continue
            result.append(c if c in "\r\n" else " ")
            i += 1
            continue

        if state == "raw_string":
            close = _raw_string_closes(text, i, raw_delimiter)
            if close is not None:
                for k in range(i, close):
                    result.append(" ")
                state = "normal"
                raw_delimiter = ""
                i = close
                continue
            result.append(c if c in "\r\n" else " ")
            i += 1
            continue

        msg = f"unexpected lexer state: {state}"
        raise RuntimeError(msg)

    return "".join(result)


def marker_mask(text: str, profile: MarkerProfile) -> str:
    """Return a same-length mask using generic line and block markers."""
    result: list[str] = []
    i = 0
    n = len(text)
    state = "normal"
    block_end = ""

    while i < n:
        c = text[i]

        if state == "normal":
            matched_block = False
            for start, end in profile.block_markers:
                if text.startswith(start, i):
                    result.extend(start)
                    i += len(start)
                    state = "block"
                    block_end = end
                    matched_block = True
                    break
            if matched_block:
                continue

            matched_line = False
            for marker in profile.line_markers:
                if text.startswith(marker, i):
                    result.extend(marker)
                    i += len(marker)
                    state = "line"
                    matched_line = True
                    break
            if matched_line:
                continue

            result.append(c if c in "\r\n" else " ")
            i += 1
            continue

        if state == "line":
            if c in "\r\n":
                state = "normal"
                result.append(c)
                if c == "\r" and i + 1 < n and text[i + 1] == "\n":
                    result.append("\n")
                    i += 2
                    continue
                i += 1
                continue
            result.append(c)
            i += 1
            continue

        if state == "block":
            if text.startswith(block_end, i):
                result.extend(block_end)
                i += len(block_end)
                state = "normal"
                continue
            result.append(c if c in "\r\n" else c)
            i += 1
            continue

        msg = f"unexpected marker lexer state: {state}"
        raise RuntimeError(msg)

    return "".join(result)


def _load_tomllib(path: str, require_toml: bool) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as e:
            if require_toml:
                msg = (
                    "tomllib or tomli are required to read comment pattern files "
                    f"but could not be imported, got: {e}"
                )
                raise CommentConfigError(msg) from None
            msg = (
                "tomllib or tomli are required to read comment pattern files "
                f"but could not be imported, got: {e}"
            )
            raise CommentConfigError(msg) from None

    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        msg = f"{path}: root must be a table"
        raise CommentConfigError(msg)
    return data


def _parse_block_markers(
    block_val: Any, profile_name: str, path: str
) -> list[tuple[str, str]]:
    if not isinstance(block_val, list):
        msg = f"{path}: profiles.{profile_name}.block must be a list"
        raise CommentConfigError(msg)
    block_markers: list[tuple[str, str]] = []
    for pair in block_val:
        if not isinstance(pair, list) or len(pair) != 2:
            msg = (
                f"{path}: profiles.{profile_name}.block entries "
                "must be two-element lists"
            )
            raise CommentConfigError(msg)
        start, end = pair
        if not isinstance(start, str) or not isinstance(end, str):
            msg = f"{path}: profiles.{profile_name}.block markers must be strings"
            raise CommentConfigError(msg)
        if not start or not end:
            msg = f"{path}: profiles.{profile_name}.block markers must be non-empty"
            raise CommentConfigError(msg)
        block_markers.append((start, end))
    return block_markers


def _resolve_marker_profile(
    name: str,
    raw_profiles: dict[str, Any],
    path: str,
    resolved: dict[str, MarkerProfile],
    chain: list[str],
) -> MarkerProfile:
    if name == CFAMILY_PROFILE:
        msg = f"{path}: cannot define built-in profile {CFAMILY_PROFILE} in profiles"
        raise CommentConfigError(msg)
    if name in chain:
        msg = f"{path}: inheritance cycle involving profile {name}"
        raise CommentConfigError(msg)
    if name in resolved:
        return resolved[name]

    if name not in raw_profiles:
        msg = f"{path}: unknown profile {name}"
        raise CommentConfigError(msg)

    defn = raw_profiles[name]
    if not isinstance(defn, dict):
        msg = f"{path}: profiles.{name} must be a table"
        raise CommentConfigError(msg)

    chain.append(name)
    line_markers: list[str] = []
    block_markers: list[tuple[str, str]] = []

    inherit = defn.get("inherit")
    if inherit is not None:
        if not isinstance(inherit, str):
            msg = f"{path}: profiles.{name}.inherit must be a string"
            raise CommentConfigError(msg)
        parent = _resolve_marker_profile(inherit, raw_profiles, path, resolved, chain)
        line_markers.extend(parent.line_markers)
        block_markers.extend(parent.block_markers)

    if "line" in defn:
        line_val = defn["line"]
        if not isinstance(line_val, list) or not all(
            isinstance(item, str) for item in line_val
        ):
            msg = f"{path}: profiles.{name}.line must be a list of strings"
            raise CommentConfigError(msg)
        line_markers.extend(line_val)

    if "block" in defn:
        block_markers.extend(_parse_block_markers(defn["block"], name, path))

    profile = MarkerProfile(
        line_markers=tuple(line_markers),
        block_markers=tuple(block_markers),
    )
    resolved[name] = profile
    chain.pop()
    return profile


def _profile_exists(name: str, raw_profiles: dict[str, Any]) -> bool:
    return name == CFAMILY_PROFILE or name in raw_profiles


def load_patterns_file(
    path: str,
    base: CommentRegistry | None = None,
) -> CommentRegistry:
    if not os.path.isfile(path):
        msg = f"cannot find comment patterns file: {path}"
        raise CommentConfigError(msg)

    data = _load_tomllib(path, require_toml=True)
    raw_extensions = data.get("extensions", {})
    raw_profiles = data.get("profiles", {})

    if not isinstance(raw_extensions, dict):
        msg = f"{path}: extensions must be a table"
        raise CommentConfigError(msg)
    if not isinstance(raw_profiles, dict):
        msg = f"{path}: profiles must be a table"
        raise CommentConfigError(msg)

    resolved_profiles: dict[str, MarkerProfile] = {}
    for profile_name in raw_profiles:
        if not isinstance(profile_name, str):
            msg = f"{path}: profile names must be strings"
            raise CommentConfigError(msg)
        _resolve_marker_profile(profile_name, raw_profiles, path, resolved_profiles, [])

    extensions: dict[str, str] = {}
    for ext, profile_name in raw_extensions.items():
        if not isinstance(ext, str) or not isinstance(profile_name, str):
            msg = f"{path}: extensions entries must be string keys and values"
            raise CommentConfigError(msg)
        if not ext.startswith("."):
            msg = f"{path}: extension {ext!r} must start with '.'"
            raise CommentConfigError(msg)
        if not _profile_exists(profile_name, raw_profiles):
            msg = f"{path}: extension {ext!r} references unknown profile {profile_name}"
            raise CommentConfigError(msg)
        extensions[ext.lower()] = profile_name

    custom_registry = CommentRegistry(
        extensions=extensions,
        marker_profiles=resolved_profiles,
    )
    if base is None:
        return merge_registries(builtin_registry(), custom_registry)
    return merge_registries(base, custom_registry)
