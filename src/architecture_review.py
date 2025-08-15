"""Architecture analysis and documentation generation tool.

This module analyzes the repository structure, builds dependency graphs,
and generates architecture documentation. It helps visualize module
dependencies and understand the codebase structure for maintenance and review.

Features:
- Parse Python module dependencies using AST analysis
- Build directed dependency graphs with NetworkX
- Generate architecture diagrams in SVG format
- Create markdown documentation of module relationships
- Analyze external dependencies from requirements files

Classes:
    ArchitectureReview: Main class for performing architecture analysis

Usage:
    python -m src.architecture_review
    
Example:
    >>> from architecture_review import ArchitectureReview
    >>> review = ArchitectureReview()
    >>> review.analyze()
    >>> review.save_diagram("architecture/diagram.svg")
    >>> review.save_summary("architecture/summary.md")

The tool automatically discovers Python files, analyzes import relationships,
and generates comprehensive documentation suitable for technical reviews
and onboarding documentation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class ArchitectureReview:
    """Analyze repository structure and create an architecture summary."""

    src_path: Path = Path("src")
    requirements_file: Path = Path("requirements.txt")
    graph: nx.DiGraph = field(init=False, default_factory=nx.DiGraph)
    dependencies: List[str] = field(init=False, default_factory=list)

    def analyze(self) -> None:
        """Populate dependency graph and requirements list."""
        self.dependencies = self._parse_requirements()
        self.graph = self._build_dependency_graph()

    def _parse_requirements(self) -> List[str]:
        if not self.requirements_file.exists():
            return []
        deps = []
        for line in self.requirements_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line.split("==")[0])
        return deps

    def _build_dependency_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        src_dir = self.src_path
        py_files = list(src_dir.glob("*.py"))
        module_names = {f.stem for f in py_files}
        for file in py_files:
            module = file.stem
            graph.add_node(module)
            tree = ast.parse(file.read_text(), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split(".")[0]
                        if name in module_names:
                            graph.add_edge(module, name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        name = node.module.split(".")[0]
                        if name in module_names:
                            graph.add_edge(module, name)
                    elif node.level >= 1:
                        # relative import from same package
                        name = node.module.split(".")[0] if node.module else module
                        if name in module_names:
                            graph.add_edge(module, name)
        return graph

    def save_diagram(self, path: str = "architecture/diagram.svg") -> None:
        """Write a dependency diagram to the given path.

        The diagram is saved in SVG format so the file is text-based and can be
        diffed easily in version control systems that do not handle binary
        files well.
        """
        if not self.graph:
            self.analyze()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray"
        )
        plt.tight_layout()
        plt.savefig(out_path, format="svg")
        plt.close()

    def save_summary(self, path: str = "architecture/architecture_review.md") -> None:
        """Write a markdown summary of the architecture analysis."""
        if not self.graph:
            self.analyze()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Architecture Review", ""]
        lines.append("## Dependencies")
        for dep in self.dependencies:
            lines.append(f"- {dep}")
        lines.append("")
        lines.append("## Module Graph")
        for node in sorted(self.graph.nodes):
            deps = sorted(self.graph.successors(node))
            if deps:
                lines.append(f"- **{node}** depends on: {', '.join(deps)}")
            else:
                lines.append(f"- **{node}** has no internal dependencies")
        out_path.write_text("\n".join(lines))


def main() -> None:
    review = ArchitectureReview()
    review.analyze()
    review.save_diagram()
    review.save_summary()


if __name__ == "__main__":
    main()
