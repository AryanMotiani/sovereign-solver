# SkilledAgent Workspace Summary Index

> **Token Optimization Rule:** This file serves as a lightweight summary index of the project's codebase, architecture, active spec, and ticket status. The AI agent MUST reference this file first before performing recursive scans of source files or full markdown specs.

---

## 1. Project Overview
- **Name:** SkilledAgent Developer Workspace
- **Type:** Developer Tooling / Agent Orchestration System
- **Core Engine:** Node.js / CLI Scaffolder
- **Main Entrypoint:** `bin/cli.js` (`npx skilledagent`)
- **Config Directory:** `.agents/`

---

## 2. Active Specification Summary
- **Current Status:** Operational / Enhanced System Architecture (v1.1.0)
- **Active Spec Reference:** `.agents/ACTIVE_SPEC.md`
- **Shared Memory Reference:** `.agents/CONTEXT.md`
- **Skill Lookup Reference:** `.agents/SKILLS_INDEX.md`

---

## 3. Architecture & Key Files Map
| Path | Component | Purpose |
|---|---|---|
| `bin/cli.js` | Scaffolder | Executable CLI that copies `.agents/` into target projects |
| `package.json` | Manifest | Package metadata, bin definition, dependencies, versioning |
| `.agents/AGENTS.MD` | Core Rules | Primary instructions, persistent prompt grilling protocol, skill routing table |
| `.agents/CONTEXT.md` | Shared Memory | Engineering standards, ubiquitous domain language, project ADRs |
| `.agents/SKILLS_INDEX.md` | Skill Index | 1-line token-optimized lookup table for all 44 available skills |
| `.agents/workflows/kickoff.md` | Workflow | 12-phase project kickoff pipeline (Scan → Vision → Big Decisions → Wayfinder → Spec → Tickets → TDD → Review) |

---

## 4. Active Backlog & Ticket Summary
- Ticket tracking mode: GitHub / Local Markdown (`.scratch/`)
- Active features: Always-active prompt grilling, token indexing, ticket-by-ticket implementation loop, top skills.sh integrations.

---

## 5. Token Guardrails
- **Max Document Limit:** 12,000 lines / ~120k tokens per document.
- **Index-First Protocol:** Agent MUST read `SUMMARY_INDEX.md` and `SKILLS_INDEX.md` before reading detailed codebase files or full skill markdowns.
