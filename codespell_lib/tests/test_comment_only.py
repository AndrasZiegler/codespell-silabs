import contextlib
import inspect
import os
import sys
import textwrap
from io import StringIO
from pathlib import Path
from collections.abc import Generator
from typing import Any

import pytest

import codespell_lib as cs_
from codespell_lib._codespell import EX_CONFIG, EX_DATAERR, EX_OK, EX_USAGE
from codespell_lib._comment_util import (
    CommentConfigError,
    c_family_mask,
    concat_fragments,
    load_patterns_file,
    repartition_mask,
)


class MainWrapper:
    @staticmethod
    def main(
        *args: Any,
        count: bool = True,
        std: bool = False,
    ) -> int | tuple[int, str, str]:
        args = tuple(str(arg) for arg in args)
        if count:
            args = ("--count", *args)
        code = cs_.main(*args)
        frame = inspect.currentframe()
        assert frame is not None
        frame = frame.f_back
        assert frame is not None
        capsys = frame.f_locals["capsys"]
        stdout, stderr = capsys.readouterr()
        assert code in (EX_OK, EX_USAGE, EX_DATAERR, EX_CONFIG)
        if code == EX_DATAERR:
            code = int(stderr.split("\n")[-2])
        elif code == EX_OK and count:
            code = int(stderr.split("\n")[-2])
            assert code == 0
        if std:
            return (code, stdout, stderr)
        return code


cs = MainWrapper()


def test_c_family_mask_line_and_block_comments() -> None:
    source = "int x; // abandonned\n/* buring */\n"
    mask = c_family_mask(source)
    assert mask == "       // abandonned\n/* buring */\n"
    assert len(mask) == len(source)


def test_c_family_mask_empty_and_comment_free() -> None:
    assert c_family_mask("") == ""
    source = "int main() { return 0; }\n"
    mask = c_family_mask(source)
    assert len(mask) == len(source)
    assert "abandonned" not in mask


def test_c_family_mask_multiple_comments_on_one_line() -> None:
    source = "int x; // a abandonned /* buring */ // c abandonned\n"
    mask = c_family_mask(source)
    assert "abandonned" in mask
    assert mask.count("abandonned") == 2


def test_c_family_mask_ignores_code_strings_and_chars() -> None:
    source = textwrap.dedent(
        """
      int clas;
      char c = 'x';
      const char *s = "opem clas // not comment";
      """
    )
    mask = c_family_mask(source)
    assert "clas" not in mask.replace(" ", "")
    assert "opem" not in mask.replace(" ", "")
    assert "not comment" not in mask


def test_c_family_mask_comment_like_text_in_strings() -> None:
    source = 'const char *s = "http://foo /* not */ ";\n'
    mask = c_family_mask(source)
    assert "http" not in mask.replace(" ", "")
    assert "not" not in mask.replace(" ", "")


def test_c_family_mask_raw_strings() -> None:
    source = 'auto s = R"delim(opem clas)delim";\n'
    mask = c_family_mask(source)
    assert "opem" not in mask.replace(" ", "")
    assert "clas" not in mask.replace(" ", "")

    multiline = 'auto s = R"xyz(\nopem\nclas\n)xyz";\n'
    mask = c_family_mask(multiline)
    assert "opem" not in mask.replace(" ", "")
    assert "clas" not in mask.replace(" ", "")


def test_c_family_mask_raw_string_unrelated_paren() -> None:
    source = 'auto s = R"()())";\n'
    mask = c_family_mask(source)
    assert mask.endswith("\n")
    assert len(mask) == len(source)


def test_c_family_mask_backslash_newline_splice() -> None:
    source = "// abandonned \\" + "\n" + "continued\n"
    mask = c_family_mask(source)
    assert "abandonned" in mask
    assert "continued" in mask

    formed = "int x = 1 /\\" + "\n" + "/abandonned\n"
    mask = c_family_mask(formed)
    assert "abandonned" in mask


def test_c_family_mask_backslash_not_splice() -> None:
    source = "int x = 1 \\ abandonned\n"
    mask = c_family_mask(source)
    assert "abandonned" not in mask.replace(" ", "")


def test_c_family_mask_crlf() -> None:
    source = "int x;\r\n// abandonned\r\n"
    mask = c_family_mask(source)
    assert "abandonned" in mask
    assert len(mask) == len(source)


def test_c_family_mask_preprocessor_and_if_zero() -> None:
    source = textwrap.dedent(
        """
      #include <stdio.h> // abandonned
      #if 0
      // buring
      #endif
      """
    )
    mask = c_family_mask(source)
    assert "abandonned" in mask
    assert "buring" in mask


def test_c_family_mask_unicode_in_comments() -> None:
    source = "// café abandonned\n"
    mask = c_family_mask(source)
    assert "café" in mask
    assert "abandonned" in mask


def test_repartition_mask() -> None:
    fragments = [
        (False, 0, ["line1\n", "line2\n"]),
        (True, 2, ["skip\n"]),
        (False, 3, ["line3\n"]),
    ]
    full_text = concat_fragments(fragments)
    full_mask = c_family_mask(full_text)
    masked = repartition_mask(fragments, full_mask)
    pos = 0
    for fragment_index, (_, _, lines) in enumerate(fragments):
        for line_index, line in enumerate(lines):
            end = pos + len(line)
            assert masked[fragment_index][2][line_index] == full_mask[pos:end]
            pos = end


def test_load_patterns_file_custom_profile(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.toml"
    patterns.write_text(
        textwrap.dedent(
            """
          [extensions]
          ".ino" = "c-family"
          ".custom" = "hash-comments"

          [profiles.hash-comments]
          line = ["#"]
          block = [["/*", "*/"]]

          [profiles.derived]
          inherit = "hash-comments"
          line = ["//"]
          """
        ),
        encoding="utf-8",
    )
    registry = load_patterns_file(str(patterns))
    assert registry.lookup_profile("firmware.ino") == "c-family"
    assert registry.lookup_profile("file.custom") == "hash-comments"

    source = "# abandonned\n"
    mask = registry.mask_text(source, "hash-comments")
    assert "abandonned" in mask


def test_load_patterns_user_override(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.toml"
    patterns.write_text(
        '[extensions]\n".c" = "hash-comments"\n\n'
        '[profiles.hash-comments]\nline = ["#"]\n',
        encoding="utf-8",
    )
    registry = load_patterns_file(str(patterns))
    source = "int x; // abandonned\n# buring\n"
    mask = registry.mask_text(source, registry.lookup_profile("foo.c") or "")
    assert "abandonned" not in mask.replace(" ", "")
    assert "buring" in mask


def test_load_patterns_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("extensions = []\n", encoding="utf-8")
    with pytest.raises(CommentConfigError, match="extensions must be a table"):
        load_patterns_file(str(bad))

    cycle = tmp_path / "cycle.toml"
    cycle.write_text(
        '[profiles.a]\ninherit = "b"\n[profiles.b]\ninherit = "a"\n',
        encoding="utf-8",
    )
    with pytest.raises(CommentConfigError, match="inheritance cycle"):
        load_patterns_file(str(cycle))


def test_comments_only_c_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text(
        textwrap.dedent(
            """
          int clas;
          /* tis abandonned */
          // buring cpu
          """
        ),
        encoding="utf-8",
    )
    assert cs.main("--comments-only", source) == 2


def test_comments_only_unsupported_extension_full_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    py_file = tmp_path / "script.py"
    py_file.write_text('clas = "abandonned"\n', encoding="utf-8")
    md_file = tmp_path / "README.md"
    md_file.write_text("abandonned clas\n", encoding="utf-8")
    assert cs.main("--comments-only", py_file) == 1
    assert cs.main("--comments-only", "--builtin", "clear,code", md_file) == 2


def test_comments_only_case_insensitive_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.C"
    source.write_text("// abandonned\nint clas;\n", encoding="utf-8")
    assert cs.main("--comments-only", source) == 1


def test_comments_only_ignore_directives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text(
        textwrap.dedent(
            """
          // abandonned codespell:ignore
          int clas; // codespell:ignore-next-line clas
          clas = 1;
          """
        ),
        encoding="utf-8",
    )
    assert cs.main("--comments-only", "--builtin", "code", source) == 0


def test_comments_only_ignore_directive_in_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text(
        "int clas; // codespell:ignore-next-line\nclas = 1;\n",
        encoding="utf-8",
    )
    assert cs.main("--comments-only", "--builtin", "code", source) == 0


def test_comments_only_ignore_multiline_regex_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text(
        textwrap.dedent(
            """
          const char *s = "opem";
          /* codespell:ignore-begin */
          int clas;
          /* codespell:ignore-end */
          // abandonned
          """
        ),
        encoding="utf-8",
    )
    args = (
        "--comments-only",
        "--ignore-multiline-regex",
        r"/\* codespell:ignore-begin.*codespell:ignore-end \*/",
        source,
    )
    assert cs.main(*args) == 1


def test_comments_only_ignore_regex_on_comments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text("// abandonned foo\nint clas;\n", encoding="utf-8")
    assert (
        cs.main(
            "--comments-only",
            "--ignore-regex",
            r"foo",
            source,
        )
        == 1
    )


def test_comments_only_rejects_write_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text("// abandonned\n", encoding="utf-8")
    result = cs.main("--comments-only", "-w", source, std=True)
    assert isinstance(result, tuple)
    code, _, stderr = result
    assert code == EX_USAGE
    assert "--comments-only" in stderr
    assert source.read_text(encoding="utf-8") == "// abandonned\n"


def test_comments_only_rejects_interactive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text("// abandonned\n", encoding="utf-8")
    result = cs.main("--comments-only", "-i", "1", source, std=True)
    assert isinstance(result, tuple)
    code, _, stderr = result
    assert code == EX_USAGE
    assert "--comments-only" in stderr


def test_comments_only_allows_ignore_words_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "code.c"
    source.write_text("// abandonned\n", encoding="utf-8")
    ignore = tmp_path / "ignore.txt"
    ignore.write_text("abandonned\n", encoding="utf-8")
    assert cs.main("--comments-only", "-I", str(ignore), source) == 0


def test_comments_only_preserves_check_filenames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "abandonned.c"
    source.write_text("// good\n", encoding="utf-8")
    assert cs.main("--comments-only", "-f", source) == 1


def test_comments_only_stdin_full_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    @contextlib.contextmanager
    def fake_stdin(text: str) -> Generator[None, None, None]:
        oldin = sys.stdin
        try:
            sys.stdin = StringIO(text)
            yield
        finally:
            sys.stdin = oldin

    with fake_stdin("clas abandonned\n"):
        result = cs.main("--comments-only", "--builtin", "clear,code", "-", std=True)
    assert isinstance(result, tuple)
    code, stdout, stderr = result
    assert code == 2
    assert "clas" in stdout
    assert "abandonned" in stdout


def test_comments_only_custom_patterns_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    patterns = tmp_path / "patterns.toml"
    patterns.write_text(
        '[extensions]\n".ino" = "c-family"\n',
        encoding="utf-8",
    )
    source = tmp_path / "sketch.ino"
    source.write_text("// abandonned\nint clas;\n", encoding="utf-8")
    assert (
        cs.main(
            "--comments-only",
            "--comment-patterns-file",
            str(patterns),
            source,
        )
        == 1
    )


def test_comments_only_missing_patterns_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = cs.main(
        "--comments-only",
        "--comment-patterns-file",
        str(tmp_path / "missing.toml"),
        std=True,
    )
    assert isinstance(result, tuple)
    code, _, stderr = result
    assert code == EX_USAGE
    assert "cannot find comment patterns file" in stderr


def test_example_code_c(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = Path(__file__).resolve().parents[2] / "example" / "code.c"
    if not example.is_file():
        pytest.skip("example/code.c not available")
    result = cs.main("--comments-only", example, std=True)
    assert isinstance(result, tuple)
    code, stdout, stderr = result
    assert code == 1
    assert "buring" in stdout
    assert "clas" not in stdout
    assert "opem" not in stdout
