"""Harness loading and artifact construction."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import HarnessArtifact, make_artifact_id

_TF_CALL_RE = re.compile(r"\btf\.[A-Za-z_][A-Za-z0-9_.]*")
_HEADER_YAML_RE = re.compile(r"^#\s*source_yaml\s*=\s*(.+?)\s*$", re.M)
_HEADER_API_RE = re.compile(r"^#\s*(?:api_name|target_api)\s*=\s*(.+?)\s*$", re.M)


class HarnessLoader:
    """Load one harness or a directory of harnesses without executing them."""

    def __init__(self, source: str = "fuzz_api_one"):
        self.source = source

    def calculate_hash(self, path: str | Path) -> str:
        p = Path(path)
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    def infer_api_name(self, path: str | Path, source: Optional[str] = None) -> str:
        p = Path(path)
        text = source if source is not None else p.read_text(encoding="utf-8", errors="replace")

        m = _HEADER_API_RE.search(text)
        if m:
            return m.group(1).strip()

        m = _HEADER_YAML_RE.search(text)
        if m:
            yaml_path = Path(m.group(1).strip())
            if yaml_path.suffix in (".yaml", ".yml"):
                return yaml_path.stem

        # fuzz_api_one names harnesses llm.<api>.py
        name = p.name
        if name.startswith("llm.") and name.endswith(".py"):
            candidate = name[len("llm."):-len(".py")]
            if candidate.startswith("tf."):
                return candidate

        # Fall back to the most plausible tf.* call in the source.
        calls = _TF_CALL_RE.findall(text)
        if calls:
            # Prefer calls that look like API entry points, not tf.constant/tf.identity.
            preferred = [
                c for c in calls
                if not c.startswith(("tf.constant", "tf.identity", "tf.convert_to_tensor", "tf.raw_ops"))
            ]
            return (preferred or calls)[0]
        return p.stem

    def load(
        self,
        harness_path: str | Path,
        psr_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
        artifact_id: Optional[str] = None,
        api_name: Optional[str] = None,
    ) -> HarnessArtifact:
        p = Path(harness_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"harness not found: {p}")
        if not p.is_file():
            raise ValueError(f"harness path is not a file: {p}")

        source = p.read_text(encoding="utf-8", errors="replace")
        inferred = api_name or self.infer_api_name(p, source)
        h = self.calculate_hash(p)

        # Header-derived PSR path is useful when --psr is not supplied.
        if psr_path is None:
            m = _HEADER_YAML_RE.search(source)
            if m:
                candidate = Path(m.group(1).strip())
                if candidate.exists():
                    psr_path = candidate
        if psr_path is not None:
            psr_path = str(Path(psr_path).resolve())

        metadata: Dict[str, Any] = {}
        if metadata_path is not None:
            metadata_path = str(Path(metadata_path).resolve())
            try:
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        return HarnessArtifact(
            artifact_id=artifact_id or make_artifact_id(inferred, h),
            framework="tensorflow",
            api_name=inferred,
            harness_path=str(p),
            harness_hash=h,
            psr_path=psr_path,
            metadata_path=metadata_path,
            source=self.source,
            tensorflow_version=metadata.get("tensorflow_version"),
            metadata=metadata,
        )

    def load_directory(
        self,
        harness_dir: str | Path,
        manifest_path: Optional[str | Path] = None,
    ) -> List[HarnessArtifact]:
        d = Path(harness_dir).resolve()
        if not d.exists():
            raise FileNotFoundError(f"harness directory not found: {d}")

        entries: List[Dict[str, Any]] = []
        if manifest_path:
            mp = Path(manifest_path).resolve()
            data = json.loads(mp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        entries.append({"api_name": key, "harness_path": value})
                    elif isinstance(value, dict):
                        entries.append({"api_name": key, **value})
            else:
                raise ValueError(f"manifest must be a list or object: {mp}")

        if not entries:
            for p in sorted(d.glob("*.py")):
                if p.name.startswith("__"):
                    continue
                entries.append({"harness_path": str(p)})

        artifacts: List[HarnessArtifact] = []
        for item in entries:
            hp = item.get("harness_path") or item.get("path") or item.get("harness")
            if not hp:
                continue
            hp_path = Path(hp)
            if not hp_path.is_absolute():
                hp_path = (d / hp_path).resolve()
            if not hp_path.exists():
                continue
            artifacts.append(
                self.load(
                    hp_path,
                    psr_path=item.get("psr") or item.get("psr_path"),
                    metadata_path=item.get("metadata") or item.get("metadata_path"),
                    api_name=item.get("api") or item.get("api_name") or item.get("harness_id"),
                )
            )
        return artifacts
