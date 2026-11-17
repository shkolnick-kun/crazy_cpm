#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace non-ASCII characters in Python docstrings and comments
with ASCII-equivalent sequences.

The script:
  * parses the source with `ast` to identify *real* docstrings
    (module/class/function) -- NOT every string literal;
  * tokenizes the source with `tokenize` to locate comments;
  * replaces every non-ASCII character in those spans using an
    explicit mapping table, falling back to NFKD transliteration
    and finally to '?'.

Usage:
    python asciify_docstrings.py file.py            # print to stdout
    python asciify_docstrings.py -i file.py         # modify in place
    python asciify_docstrings.py -n file.py         # show unified diff
"""

import argparse
import ast
import difflib
import io
import sys
import tokenize
import unicodedata


# ---------------------------------------------------------------------------
# Explicit mapping for typographic / mathematical symbols.
# These are the characters that actually appear in English-language source.
# ---------------------------------------------------------------------------
REPLACEMENTS = {
    # Dashes and hyphens
    '\u2014': '--',        # — em dash
    '\u2013': '-',         # – en dash
    '\u2212': '-',         # − minus sign
    '\u2010': '-',         # ‐ hyphen
    '\u2011': '-',         # ‑ non-breaking hyphen
    '\u2012': '-',         # ‒ figure dash
    '\u00ad': '-',         # soft hyphen
    # Arrows
    '\u2192': '->',        # →
    '\u2190': '<-',        # ←
    '\u2194': '<->',       # ↔
    '\u21d2': '=>',        # ⇒
    '\u21d0': '<=',        # ⇐
    '\u21d4': '<=>',       # ⇔
    # Comparisons / math
    '\u2265': '>=',        # ≥
    '\u2264': '<=',        # ≤
    '\u2260': '!=',        # ≠
    '\u2248': '~=',        # ≈
    '\u2245': '~=',        # ≅
    '\u2261': '==',        # ≡
    '\u00b1': '+/-',       # ±
    '\u00d7': 'x',         # ×
    '\u00f7': '/',         # ÷
    '\u221d': 'prop',      # ∝
    '\u221e': 'inf',       # ∞
    '\u221a': 'sqrt',      # √
    '\u2211': 'Sum',       # ∑
    '\u220f': 'Prod',      # ∏
    '\u222b': 'Int',       # ∫
    '\u2205': 'empty',     # ∅
    '\u2208': 'in',        # ∈
    '\u2209': 'not in',    # ∉
    '\u2282': 'subset',    # ⊂
    '\u2283': 'superset',  # ⊃
    '\u2200': 'forall',    # ∀
    '\u2203': 'exists',    # ∃
    # Punctuation / typography
    '\u00b0': 'deg',       # °
    '\u00b7': '.',         # ·
    '\u2026': '...',       # …
    '\u2022': '*',         # •
    '\u25e6': 'o',         # ◦
    '\u2032': "'",         # ′
    '\u2033': '"',         # ″
    '\u201c': '"',         # “
    '\u201d': '"',         # ”
    '\u2018': "'",         # ‘
    '\u2019': "'",         # ’
    '\u00ab': '<<',        # «
    '\u00bb': '>>',        # »
    '\u2039': '<',         # ‹
    '\u203a': '>',         # ›
    '\u2122': '(TM)',      # ™
    '\u00ae': '(R)',       # ®
    '\u00a9': '(C)',       # ©
    '\u00a7': 'section',   # §
    '\u00b6': 'P',         # ¶
    # Fractions
    '\u00bd': '1/2',
    '\u00bc': '1/4',
    '\u00be': '3/4',
    '\u2044': '/',         # ⁄ fraction slash
    # Spaces
    '\u00a0': ' ',         # non-breaking space
    '\u2007': ' ',         # figure space
    '\u2009': ' ',         # thin space
    '\u200a': ' ',         # hair space
    '\u202f': ' ',         # narrow no-break space
    # Superscripts / subscripts
    '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
    '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
    '\u2078': '8', '\u2079': '9',
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
    '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
    '\u2088': '8', '\u2089': '9',
    # Greek letters (lower)
    '\u03b1': 'alpha',   '\u03b2': 'beta',   '\u03b3': 'gamma',
    '\u03b4': 'delta',   '\u03b5': 'epsilon','\u03b6': 'zeta',
    '\u03b7': 'eta',     '\u03b8': 'theta',  '\u03b9': 'iota',
    '\u03ba': 'kappa',   '\u03bb': 'lambda', '\u03bc': 'mu',
    '\u03bd': 'nu',      '\u03be': 'xi',     '\u03bf': 'omicron',
    '\u03c0': 'pi',      '\u03c1': 'rho',    '\u03c3': 'sigma',
    '\u03c2': 'sigma',   '\u03c4': 'tau',    '\u03c5': 'upsilon',
    '\u03c6': 'phi',     '\u03c7': 'chi',    '\u03c8': 'psi',
    '\u03c9': 'omega',
    # Greek letters (upper)
    '\u0391': 'Alpha',   '\u0392': 'Beta',   '\u0393': 'Gamma',
    '\u0394': 'Delta',   '\u0395': 'Epsilon','\u0396': 'Zeta',
    '\u0397': 'Eta',     '\u0398': 'Theta',  '\u0399': 'Iota',
    '\u039a': 'Kappa',   '\u039b': 'Lambda', '\u039c': 'Mu',
    '\u039d': 'Nu',      '\u039e': 'Xi',     '\u039f': 'Omicron',
    '\u03a0': 'Pi',      '\u03a1': 'Rho',    '\u03a3': 'Sigma',
    '\u03a4': 'Tau',     '\u03a5': 'Upsilon','\u03a6': 'Phi',
    '\u03a7': 'Chi',     '\u03a8': 'Psi',    '\u03a9': 'Omega',
}


def _ascii_fallback(ch: str) -> str:
    """Best-effort ASCII transliteration for a single character."""
    if ord(ch) < 128:
        return ch
    # NFKD decomposes accented Latin letters: é -> e + combining accent
    nfkd = unicodedata.normalize('NFKD', ch)
    stripped = ''.join(c for c in nfkd if not unicodedata.combining(c))
    if stripped and stripped != ch and all(ord(c) < 128 for c in stripped):
        return stripped
    return '?'


def replace_non_ascii(text: str) -> str:
    """Replace every non-ASCII character in *text* with an ASCII sequence."""
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append(REPLACEMENTS.get(ch, _ascii_fallback(ch)))
    return ''.join(out)


# ---------------------------------------------------------------------------
# Locating docstrings (via AST) and comments (via tokenize)
# ---------------------------------------------------------------------------
def _docstring_positions(source: str):
    """Return set of (lineno, col_offset) of every docstring Constant."""
    tree = ast.parse(source)
    positions = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, 'body', None)
            if body and isinstance(body[0], ast.Expr):
                val = body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    positions.add((val.lineno, val.col_offset))
    return positions


def transform(source: str) -> str:
    """Return *source* with non-ASCII chars in docstrings/comments
    replaced by ASCII equivalents."""
    try:
        docstrings = _docstring_positions(source)
    except SyntaxError as e:
        raise RuntimeError(f"Cannot parse source: {e}") from e

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

    # Line-offset table for absolute-character addressing
    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def abs_offset(pos):
        line_no, col = pos
        if line_no - 1 >= len(line_offsets):
            return len(source)
        return line_offsets[line_no - 1] + col

    # Collect non-overlapping (start, end, replacement) spans
    spans = []
    for tok in tokens:
        tok_type, tok_string, start, end, _ = tok
        if tok_type == tokenize.COMMENT:
            pass                                    # all comments
        elif tok_type == tokenize.STRING:
            if start not in docstrings:
                continue                            # not a docstring
        else:
            continue

        start_abs = abs_offset(start)
        end_abs = abs_offset(end)
        old = source[start_abs:end_abs]
        new = replace_non_ascii(old)
        if new != old:
            spans.append((start_abs, end_abs, new))

    # Apply in reverse order so earlier offsets stay valid
    result = source
    for start_abs, end_abs, new in sorted(spans, reverse=True):
        result = result[:start_abs] + new + result[end_abs:]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Replace non-ASCII characters in Python docstrings '
                    'and comments with ASCII equivalents.'
    )
    parser.add_argument('files', nargs='+', help='Python source files')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-i', '--in-place', action='store_true',
                      help='Modify files in place (default: print to stdout)')
    mode.add_argument('-n', '--dry-run', action='store_true',
                      help='Show unified diff without modifying anything')
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                src = f.read()
            new = transform(src)
        except Exception as e:                      # noqa: BLE001
            print(f"ERROR: {path}: {e}", file=sys.stderr)
            exit_code = 1
            continue

        if new == src:
            print(f"No changes: {path}", file=sys.stderr)
            continue

        if args.dry_run:
            sys.stdout.writelines(difflib.unified_diff(
                src.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=path, tofile=path,
            ))
        elif args.in_place:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new)
            print(f"Updated: {path}", file=sys.stderr)
        else:
            sys.stdout.write(new)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())