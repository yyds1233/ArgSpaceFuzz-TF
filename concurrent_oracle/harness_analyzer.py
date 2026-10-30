"""Static AST analysis of generated TensorFlow harnesses."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import HarnessAnalysis, HarnessArtifact, TargetCall

_TENSOR_CTORS = {
    "tf.constant", "tf.convert_to_tensor", "tf.zeros", "tf.ones", "tf.fill",
    "tf.random.normal", "tf.random.uniform", "tf.random.stateless_normal",
    "tf.random.stateless_uniform", "tf.reshape", "tf.cast", "tf.stack",
    "tf.concat", "tf.Variable",
}
_RESOURCE_CTORS = {
    "tf.Variable", "tf.data.Dataset", "tf.compat.v1.Session",
    "tf.compat.v1.data.Dataset", "tf.data.Iterator",
}
_CALLBACK_SINKS = {
    "tf.py_function", "tf.numpy_function", "tf.data.Dataset.from_generator",
    "tf.data.Dataset.map", "tf.data.Dataset.interleave", "tf.data.Dataset.flat_map",
}
_DATASET_METHODS = {
    "map", "interleave", "prefetch", "batch", "shuffle", "repeat", "cache",
    "from_tensor_slices", "from_generator", "from_tensors", "as_numpy_iterator",
    "make_one_shot_iterator", "get_next", "take", "skip", "flat_map",
}
_CLEANUP_METHODS = {"close", "__exit__", "clear_session", "reset_default_graph", "terminate"}


def dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


def _segment(node: ast.AST, source: str) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if seg:
            return " ".join(seg.split())
    except Exception:
        pass
    try:
        return ast.unparse(node)
    except Exception:
        return "<expression>"


def _assigned_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            names.extend(_target_names(target))
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        names.extend(_target_names(node.target))
    return names


def _target_names(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: List[str] = []
        for elt in node.elts:
            out.extend(_target_names(elt))
        return out
    return []


def _target_match(call_name: str, api_name: str) -> bool:
    if not call_name or not api_name:
        return False
    if call_name == api_name:
        return True
    # tf.math.log1p may be exercised through tf.raw_ops.Log1p.
    last = api_name.split(".")[-1].lower()
    if call_name.lower() == f"tf.raw_ops.{last}":
        return True
    # Composite APIs may call an underlying op with the same leaf name.
    if call_name.startswith("tf.") and call_name.lower().split(".")[-1] == last:
        return True
    return False


class HarnessAnalyzer:
    """Extract the structure needed by suitability rules and transformation."""

    def analyze(self, artifact: HarnessArtifact) -> HarnessAnalysis:
        p = Path(artifact.harness_path)
        source = p.read_text(encoding="utf-8", errors="replace")
        result = HarnessAnalysis(artifact_id=artifact.artifact_id)
        try:
            tree = ast.parse(source, filename=str(p))
        except SyntaxError as exc:
            result.ast_parse_status = "FAILED"
            result.warnings.append(f"SyntaxError: {exc}")
            return result

        result.imports = self.extract_imports(tree)
        result.target_calls = self.extract_target_calls(source, tree, artifact.api_name)
        result.input_variables = self.extract_input_variables(tree)
        result.resource_variables = self.extract_resource_variables(tree)
        result.callback_functions = self.extract_callback_functions(tree)
        result.output_expressions = self.extract_output_expressions(source, tree)
        result.cleanup_expressions = self.extract_cleanup_expressions(source, tree)
        result.tensor_construction = self.extract_tensor_construction(source, tree)
        result.dataset_operations = self.extract_dataset_operations(source, tree)
        result.resource_lifecycle = self.extract_resource_lifecycle(source, tree)

        if not result.target_calls:
            result.warnings.append(f"no target call matching {artifact.api_name!r} found by AST")
        dynamic = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in {"getattr", "eval", "exec"}
        ]
        if dynamic:
            result.warnings.append("dynamic calls (getattr/eval/exec) detected; AST may be incomplete")
        return result

    # ── individual extractors ──────────────────────────────────

    def extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.append({"line": node.lineno, "module": alias.name,
                                "alias": alias.asname, "kind": "import"})
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    out.append({"line": node.lineno, "module": node.module or "",
                                "name": alias.name, "alias": alias.asname,
                                "kind": "from"})
        return out

    def extract_target_calls(self, source: str, tree: ast.AST, api_name: str) -> List[TargetCall]:
        out: List[TargetCall] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = dotted_name(node.func)
            if not name:
                continue
            if _target_match(name, api_name):
                out.append(TargetCall(
                    api_name=name,
                    line=node.lineno,
                    call_expression=_segment(node, source),
                    is_target=True,
                ))
        out.sort(key=lambda x: x.line)
        return out

    def extract_input_variables(self, tree: ast.AST) -> List[str]:
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    name = dotted_name(value.func) or ""
                    if name in _TENSOR_CTORS or name.startswith("tf.random."):
                        names.extend(_assigned_names(node))
                elif isinstance(value, ast.BinOp):
                    names.extend(_assigned_names(node))
        return sorted(set(names))

    def extract_resource_variables(self, tree: ast.AST) -> List[str]:
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = dotted_name(node.value.func) or ""
                if name in _RESOURCE_CTORS or name.startswith("tf.data.") or name == "iter":
                    names.extend(_assigned_names(node))
        return sorted(set(names))

    def extract_callback_functions(self, tree: ast.AST) -> List[str]:
        names: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            sink = dotted_name(node.func) or ""
            if sink not in _CALLBACK_SINKS:
                continue
            for arg in node.args[:2]:
                if isinstance(arg, ast.Name):
                    names.append(arg.id)
        # Also include top-level functions referenced by name in the source.
        return sorted(set(names))

    def extract_output_expressions(self, source: str, tree: ast.AST) -> List[str]:
        out: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted_name(node.func) or ""
                if name.endswith(".numpy") or name.endswith(".as_numpy_iterator") or name in {"tf.identity", "print"}:
                    out.append(_segment(node, source))
            elif isinstance(node, ast.Assert):
                out.append(_segment(node, source))
        return out

    def extract_cleanup_expressions(self, source: str, tree: ast.AST) -> List[str]:
        out: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted_name(node.func) or ""
                if name.split(".")[-1] in _CLEANUP_METHODS:
                    out.append(_segment(node, source))
        return out

    def extract_tensor_construction(self, source: str, tree: ast.AST) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted_name(node.func) or ""
                if name in _TENSOR_CTORS or name.startswith("tf.random."):
                    out.append({
                        "line": node.lineno,
                        "call": name,
                        "expression": _segment(node, source),
                        "assigned_to": _assigned_names(node) if isinstance(node, ast.Assign) else [],
                    })
        return out

    def extract_dataset_operations(self, source: str, tree: ast.AST) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted_name(node.func) or ""
                leaf = name.split(".")[-1]
                if "tf.data" in name or leaf in _DATASET_METHODS:
                    out.append({
                        "line": node.lineno,
                        "call": name,
                        "expression": _segment(node, source),
                    })
        return out

    def extract_resource_lifecycle(self, source: str, tree: ast.AST) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = dotted_name(node.func) or ""
                leaf = name.split(".")[-1]
                if name in _RESOURCE_CTORS or leaf in _CLEANUP_METHODS:
                    out.append({
                        "line": node.lineno,
                        "call": name,
                        "expression": _segment(node, source),
                    })
        return out
