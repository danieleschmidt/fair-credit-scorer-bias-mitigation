# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for this project. ADRs are documents that capture important architectural decisions made during the project, including the context, decision, and consequences.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures a single architecture decision and its rationale. It describes the forces that influence the decision and the reasoning behind the chosen solution.

## ADR Process

1. **Identify Decision**: When an architectural decision needs to be made, create a new ADR
2. **Use Template**: Copy the template.md file and rename it with the next sequential number
3. **Fill Content**: Complete all sections with relevant information
4. **Review & Approve**: Get team review before marking as "Accepted"
5. **Update Status**: Keep ADR status current (Proposed → Accepted → Superseded/Deprecated)

## ADR Statuses

- **Proposed**: Under discussion, not yet decided
- **Accepted**: Decision made and approved
- **Superseded**: Replaced by a newer decision
- **Deprecated**: No longer relevant but kept for historical context

## Current ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [0001](0001-adr-framework.md) | ADR Framework | Accepted | 2025-01-27 |
| [0002](0002-fairness-metrics-library.md) | Fairness Metrics Library Choice | Accepted | 2025-01-27 |
| [0003](0003-bias-mitigation-approach.md) | Bias Mitigation Approach | Accepted | 2025-01-27 |
| [0004](0004-devops-toolchain.md) | DevOps Toolchain Selection | Accepted | 2025-01-27 |

## Guidelines

- Keep ADRs concise but complete
- Focus on the *why* not just the *what*
- Include relevant context and constraints
- Document alternatives considered
- Update status when decisions change
- Reference related ADRs when applicable

## Resources

- [ADR GitHub Organization](https://adr.github.io/)
- [ADR Tools](https://github.com/npryce/adr-tools)
- [Template Documentation](template.md)