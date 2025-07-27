# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) that document important architectural decisions made for the Fair Credit Scorer project.

## Format

We use the template format proposed by Michael Nygard with the following structure:

- **Title**: A short phrase describing the decision
- **Status**: Proposed, Accepted, Deprecated, or Superseded
- **Context**: The situation and forces driving the decision
- **Decision**: The chosen approach and rationale
- **Consequences**: The positive and negative impacts

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-use-architecture-decision-records.md) | Use Architecture Decision Records | Accepted | 2024-07-27 |
| [0002](0002-python-as-primary-language.md) | Python as Primary Language | Accepted | 2024-07-27 |
| [0003](0003-fairlearn-for-bias-mitigation.md) | Fairlearn for Bias Mitigation | Accepted | 2024-07-27 |
| [0004](0004-docker-containerization.md) | Docker for Containerization | Accepted | 2024-07-27 |
| [0005](0005-pytest-testing-framework.md) | Pytest as Testing Framework | Accepted | 2024-07-27 |
| [0006](0006-mkdocs-documentation.md) | MkDocs for Documentation | Accepted | 2024-07-27 |

## Creating New ADRs

1. Copy `template.md` to a new file with the next sequential number
2. Fill in the template with decision details
3. Update this README with the new ADR entry
4. Submit as part of your pull request

## Guidelines

- Keep ADRs focused on architectural decisions, not implementation details
- Write in simple, clear language
- Include the business context that drove the decision
- Be honest about trade-offs and consequences
- Review and update ADRs as decisions evolve