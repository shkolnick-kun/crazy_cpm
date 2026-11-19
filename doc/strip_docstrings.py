#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strip_docstrings.py — удаление докстрингов из Python-файла.

Использование:
    python strip_docstrings.py input.py output.py
    python strip_docstrings.py input.py            # перезапись на месте
    python strip_docstrings.py input.py -          # вывод в stdout

Работает через AST: находит все узлы, у которых есть docstring
(Module, ClassDef, FunctionDef, AsyncFunctionDef), запоминает
(lineno, end_lineno) их строковых литералов и удаляет эти строки
из исходника. Также удаляет строку, оставшуюся пустой после
удаления (чтобы не копились пустые строки).
"""

import ast
import sys
import io


_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _docstring_node(node):
    """Вернуть узел docstring для node или None."""
    body = getattr(node, 'body', None)
    if not isinstance(body, (list, tuple)) or not body:
        return None
    first = body[0]
    if (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return first
    return None


def find_docstrings(tree):
    """Собрать список (lineno, end_lineno, col_offset, end_col_offset, node_type)."""
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        ds = _docstring_node(node)
        if ds is not None:
            val = ds.value
            result.append((
                val.lineno,
                val.end_lineno,
                val.col_offset,
                val.end_col_offset,
                type(node).__name__,
            ))
    return result


def strip_docstrings(source: str) -> str:
    """Удалить докстринги из исходного кода."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SystemExit(f"SyntaxError: {e}")

    docstrings = find_docstrings(tree)
    if not docstrings:
        return source

    # Разбиваем исходник на строки, сохраняя информацию о переводах строк
    lines = source.splitlines(keepends=True)

    # Для каждого докстринга определяем диапазон строк, которые надо удалить.
    # Обычно докстринг занимает одну или несколько строк. Если он занимает
    # ровно одну строку и перед ним ничего нет (только отступ), удаляем
    # строку целиком. Если докстринг многострочный — удаляем все его строки.
    to_remove = set()

    for lineno, end_lineno, col, end_col, node_type in docstrings:
        # Проверяем, что перед докстрингом на первой строке только пробелы
        first_line = lines[lineno - 1]
        prefix = first_line[:col]
        if prefix.strip() != '':
            # Докстринг не в начале строки (например, `def f(): "doc"` —
            # но это не docstring по определению, AST так не даст;
            # либо это `x = "..."` — не docstring). Пропускаем.
            continue

        # Определяем, что после докстринга на последней строке только
        # пробелы/перевод строки. Если нет — оставляем как есть (не
        # трогаем строку), чтобы не сломать код после docstring на той же
        # строке. Такое почти не встречается, но обработаем.
        last_line = lines[end_lineno - 1]
        suffix = last_line[end_col:]
        if suffix.strip() != '' and not suffix.strip().startswith('#'):
            continue

        for ln in range(lineno, end_lineno + 1):
            to_remove.add(ln - 1)  # 0-based

        # Если докстринг был единственным содержимым тела (например,
        # `def f():\n    "doc"\n`), после удаления останется пустое тело.
        # Вставляем `pass`, чтобы код остался валидным.
        # Проверяем: является ли родитель FunctionDef/ClassDef, и был ли
        # docstring единственным оператором.
        # (Простая эвристика: если это FunctionDef/ClassDef и его body
        # состоит только из docstring.)
        # Для этого нужно найти родителя — сделаем отдельно ниже.

    # Определяем, какие функции/классы останутся с пустым телом
    empty_bodies = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = _docstring_node(node)
            if ds is not None and len(node.body) == 1:
                # Тело состояло только из docstring
                empty_bodies.add(id(node))

    # Собираем индексы первых строк тел, которым нужен pass
    # (первая строка после docstring — вставляем `pass` с отступом)
    insert_pass_at = {}  # lineno (0-based) -> indent string
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ds = _docstring_node(node)
            if ds is not None and len(node.body) == 1:
                # Вставляем pass после удалённого docstring
                # Строка для вставки — end_lineno docstring (0-based)
                insert_pass_at[ds.value.end_lineno - 1] = ' ' * (node.col_offset + 4)

    # Формируем результат
    out = []
    for i, line in enumerate(lines):
        if i in insert_pass_at:
            # Эта строка — последняя строка удаляемого docstring;
            # после удаления вставим pass
            out.append(insert_pass_at[i] + 'pass\n')
            continue
        if i in to_remove:
            continue
        out.append(line)

    return ''.join(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    inp = argv[1]
    with open(inp, 'r', encoding='utf-8') as f:
        src = f.read()

    result = strip_docstrings(src)

    if len(argv) >= 3:
        outp = argv[2]
        if outp == '-':
            sys.stdout.write(result)
        else:
            with open(outp, 'w', encoding='utf-8') as f:
                f.write(result)
    else:
        with open(inp, 'w', encoding='utf-8') as f:
            f.write(result)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
