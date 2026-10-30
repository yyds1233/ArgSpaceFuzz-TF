"""API contract registry for concurrency oracle applicability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .constants import CONTRACT_STATUSES


DEFAULT_CONTRACT = {
    "api_name": "",
    "scenario": None,
    "status": "UNKNOWN",
    "source": "none",
    "properties": [],
    "notes": [],
    "last_verified_version": None,
    "oracles": [],
}


class ContractRegistry:
    def __init__(self, contract_dir: Optional[str | Path] = None):
        self._contracts: List[Dict[str, Any]] = []
        if contract_dir:
            self.load_directory(contract_dir)

    def load_directory(self, contract_dir: str | Path) -> int:
        d = Path(contract_dir)
        if not d.exists():
            return 0
        count = 0
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self.register_contract(item)
                        count += 1
            elif isinstance(data, dict):
                if "contracts" in data and isinstance(data["contracts"], list):
                    for item in data["contracts"]:
                        if isinstance(item, dict):
                            self.register_contract(item)
                            count += 1
                elif "api_name" in data:
                    self.register_contract(data)
                    count += 1
                else:
                    for api, item in data.items():
                        if isinstance(item, dict):
                            self.register_contract({"api_name": api, **item})
                            count += 1
        return count

    def register_contract(self, contract: Dict[str, Any]) -> None:
        if not isinstance(contract, dict) or not contract.get("api_name"):
            raise ValueError("contract must be a dict with api_name")
        status = str(contract.get("status", "UNKNOWN")).upper()
        if status not in CONTRACT_STATUSES:
            raise ValueError(f"invalid contract status: {status}")
        item = dict(DEFAULT_CONTRACT)
        item.update(contract)
        item["status"] = status
        item["properties"] = list(item.get("properties") or [])
        item["notes"] = list(item.get("notes") or [])
        item["oracles"] = list(item.get("oracles") or [])
        self._contracts.append(item)

    def get_contract(self, api_name: str, scenario: Optional[str] = None) -> Dict[str, Any]:
        candidates = [c for c in self._contracts if c.get("api_name") == api_name]
        if scenario:
            exact = [c for c in candidates if c.get("scenario") == scenario]
            if exact:
                return dict(exact[-1])
            generic = [c for c in candidates if not c.get("scenario")]
            if generic:
                return dict(generic[-1])
            return {**DEFAULT_CONTRACT, "api_name": api_name, "scenario": scenario}
        if candidates:
            return dict(candidates[-1])
        return {**DEFAULT_CONTRACT, "api_name": api_name, "scenario": scenario}

    def validate_oracle_applicability(self, api_name: str, scenario: str, oracle_id: str) -> Dict[str, Any]:
        contract = self.get_contract(api_name, scenario)
        status = contract.get("status", "UNKNOWN")
        allowed = contract.get("oracles") or []
        if status == "UNSUPPORTED":
            return {
                "applicable": False,
                "contract_status": status,
                "reason": f"contract marks {api_name}/{scenario} as UNSUPPORTED",
                "contract": contract,
            }
        if allowed and oracle_id not in allowed:
            return {
                "applicable": False,
                "contract_status": status,
                "reason": f"oracle {oracle_id} is not listed in the contract's allowed oracles",
                "contract": contract,
            }
        return {
            "applicable": True,
            "contract_status": status,
            "reason": "contract unknown; keep result inconclusive if evidence is incomplete"
            if status == "UNKNOWN" else "contract supported",
            "contract": contract,
        }

    def aggregate_status(self, api_name: str, scenarios: Iterable[str]) -> str:
        statuses = [self.get_contract(api_name, s).get("status", "UNKNOWN") for s in scenarios]
        if not statuses:
            return "UNKNOWN"
        if any(s == "SUPPORTED" for s in statuses):
            return "SUPPORTED"
        if all(s == "UNSUPPORTED" for s in statuses):
            return "UNSUPPORTED"
        return "UNKNOWN"

    def all_contracts(self) -> List[Dict[str, Any]]:
        return [dict(c) for c in self._contracts]
