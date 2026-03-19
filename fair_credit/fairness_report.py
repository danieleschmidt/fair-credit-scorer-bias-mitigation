"""FairnessReport: generate a structured fairness audit report.

Produces a human-readable text report and optionally a machine-readable
dict/JSON export summarising BiasDetector results before and after mitigation.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional


# Fairness thresholds (common industry / academic guidelines)
_THRESHOLD_DP_DIFF = 0.10   # |demographic parity diff| ≤ 0.10 → pass
_THRESHOLD_EO_DIFF = 0.10   # |equalized odds diff| ≤ 0.10 → pass
_THRESHOLD_IF_SCORE = 0.80  # individual fairness score ≥ 0.80 → pass


class FairnessReport:
    """Build and render a fairness audit report.

    Parameters
    ----------
    model_name : str
        Identifier for the model being audited.
    protected_attribute_name : str
        Human-readable label for the protected attribute.

    Usage
    -----
    >>> report = FairnessReport("LogisticRegression", "gender")
    >>> report.add_baseline(detector_results_before)
    >>> report.add_mitigated("reweight", detector_results_after_reweight)
    >>> report.add_mitigated("threshold", detector_results_after_threshold)
    >>> print(report.render())
    >>> report.save("fairness_audit.txt")
    """

    def __init__(
        self,
        model_name: str = "Model",
        protected_attribute_name: str = "protected_attribute",
    ) -> None:
        self.model_name = model_name
        self.protected_attribute_name = protected_attribute_name
        self._baseline: Optional[Dict[str, Any]] = None
        self._mitigated: List[Dict[str, Any]] = []
        self._timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_baseline(self, results: Dict[str, Any]) -> None:
        """Register BiasDetector results for the un-mitigated model."""
        self._baseline = dict(results)

    def add_mitigated(self, strategy_name: str, results: Dict[str, Any]) -> None:
        """Register BiasDetector results after a mitigation strategy."""
        self._mitigated.append({"strategy": strategy_name, "results": dict(results)})

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return the full audit report as a formatted string."""
        if self._baseline is None:
            return "[FairnessReport] No baseline results — call add_baseline() first."

        sections: List[str] = []
        sections.append(self._header())
        sections.append(self._baseline_section())
        for m in self._mitigated:
            sections.append(self._mitigated_section(m["strategy"], m["results"]))
        sections.append(self._summary_section())
        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Return the report data as a nested dictionary."""
        return {
            "model_name": self.model_name,
            "protected_attribute": self.protected_attribute_name,
            "timestamp": self._timestamp,
            "baseline": self._baseline,
            "mitigated": self._mitigated,
            "pass_fail": self._pass_fail(self._baseline) if self._baseline else {},
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the report data as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Write the rendered text report to *path*."""
        with open(path, "w") as f:
            f.write(self.render())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _header(self) -> str:
        return textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════╗
        ║            FAIRNESS AUDIT REPORT                    ║
        ╚══════════════════════════════════════════════════════╝
        Model               : {self.model_name}
        Protected attribute : {self.protected_attribute_name}
        Generated at        : {self._timestamp} (UTC)
        """).strip()

    def _metrics_block(self, r: Dict[str, Any], indent: int = 2) -> str:
        pad = " " * indent
        lines = [
            f"{pad}Overall accuracy              : {r.get('overall_accuracy', float('nan')):.4f}",
            f"{pad}Demographic parity difference : {r.get('demographic_parity_difference', float('nan')):.4f}",
            f"{pad}Demographic parity ratio      : {r.get('demographic_parity_ratio', float('nan')):.4f}",
            f"{pad}Equalized odds difference     : {r.get('equalized_odds_difference', float('nan')):.4f}",
            f"{pad}  TPR difference              : {r.get('tpr_difference', float('nan')):.4f}",
            f"{pad}  FPR difference              : {r.get('fpr_difference', float('nan')):.4f}",
            f"{pad}Equal opportunity difference  : {r.get('equal_opportunity_difference', float('nan')):.4f}",
        ]
        if "individual_fairness_score" in r:
            lines.append(
                f"{pad}Individual fairness score     : {r['individual_fairness_score']:.4f}"
            )
        # per-group
        if "per_group" in r:
            lines.append(f"{pad}Per-group metrics:")
            for g, m in r["per_group"].items():
                lines.append(
                    f"{pad}  Group {g}: "
                    f"acc={m['accuracy']:.3f}  sel_rate={m['selection_rate']:.3f}"
                    f"  TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}"
                    f"  F1={m['f1']:.3f}"
                )
        return "\n".join(lines)

    def _pass_fail(self, r: Dict[str, Any]) -> Dict[str, str]:
        pf: Dict[str, str] = {}
        dp = r.get("demographic_parity_difference", float("nan"))
        eo = r.get("equalized_odds_difference", float("nan"))
        ifs = r.get("individual_fairness_score")

        pf["demographic_parity"] = "PASS" if dp <= _THRESHOLD_DP_DIFF else "FAIL"
        pf["equalized_odds"] = "PASS" if eo <= _THRESHOLD_EO_DIFF else "FAIL"
        if ifs is not None:
            pf["individual_fairness"] = "PASS" if ifs >= _THRESHOLD_IF_SCORE else "FAIL"
        return pf

    def _baseline_section(self) -> str:
        pf = self._pass_fail(self._baseline)  # type: ignore[arg-type]
        lines = [
            "",
            "──────────────────────────────────────────────────────",
            "BASELINE (no mitigation)",
            "──────────────────────────────────────────────────────",
            self._metrics_block(self._baseline),  # type: ignore[arg-type]
            "",
            "  Fairness checks:",
        ]
        for criterion, verdict in pf.items():
            lines.append(f"    {criterion:<30} [{verdict}]")
        return "\n".join(lines)

    def _mitigated_section(self, strategy: str, r: Dict[str, Any]) -> str:
        pf = self._pass_fail(r)
        lines = [
            "",
            "──────────────────────────────────────────────────────",
            f"AFTER MITIGATION — {strategy.upper()}",
            "──────────────────────────────────────────────────────",
            self._metrics_block(r),
            "",
            "  Fairness checks:",
        ]
        for criterion, verdict in pf.items():
            lines.append(f"    {criterion:<30} [{verdict}]")

        # Improvement delta vs baseline
        if self._baseline is not None:
            lines.append("")
            lines.append("  Improvement vs baseline:")
            for key in (
                "demographic_parity_difference",
                "equalized_odds_difference",
                "equal_opportunity_difference",
            ):
                before = self._baseline.get(key, float("nan"))
                after = r.get(key, float("nan"))
                delta = before - after
                sign = "↓" if delta >= 0 else "↑"
                lines.append(f"    {key:<40}: {sign} {abs(delta):.4f}")
        return "\n".join(lines)

    def _summary_section(self) -> str:
        lines = [
            "",
            "══════════════════════════════════════════════════════",
            "RECOMMENDATION SUMMARY",
            "══════════════════════════════════════════════════════",
        ]
        # Find best strategy
        if self._mitigated:
            scored = []
            for m in self._mitigated:
                r = m["results"]
                score = (
                    r.get("demographic_parity_difference", 1.0)
                    + r.get("equalized_odds_difference", 1.0)
                )
                scored.append((score, m["strategy"]))
            scored.sort()
            best_score, best_strategy = scored[0]
            lines.append(f"  Best strategy : {best_strategy} (combined bias score {best_score:.4f})")
        else:
            lines.append("  No mitigation strategies evaluated yet.")

        lines += [
            "",
            f"  Thresholds used:",
            f"    Demographic parity difference ≤ {_THRESHOLD_DP_DIFF}",
            f"    Equalized odds difference     ≤ {_THRESHOLD_EO_DIFF}",
            f"    Individual fairness score     ≥ {_THRESHOLD_IF_SCORE}",
            "",
        ]
        return "\n".join(lines)
