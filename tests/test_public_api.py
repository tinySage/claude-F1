"""Every public function and method in ``f1strat`` has type hints and a docstring."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "f1strat"


def _public_functions():
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                yield path.name, node.name, node
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                        yield path.name, f"{node.name}.{sub.name}", sub


def test_public_functions_have_type_hints_and_docstrings():
    missing = []
    for module, name, fn in _public_functions():
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        untyped = [a.arg for a in args if a.arg not in ("self", "cls") and a.annotation is None]
        if untyped:
            missing.append(f"{module}:{name} untyped {untyped}")
        if fn.returns is None:
            missing.append(f"{module}:{name} no return type")
        if ast.get_docstring(fn) is None:
            missing.append(f"{module}:{name} no docstring")
    assert not missing, "\n".join(missing)
