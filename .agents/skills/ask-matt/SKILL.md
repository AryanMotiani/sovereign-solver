---
name: ask-matt
description: Ask which skill or flow fits your situation. A router over the 44 skills in this repo.
disable-model-invocation: true
---

# Ask Matt — SkilledAgent Router

You don't remember every skill, so ask. Consult `.agents/SKILLS_INDEX.md` for fast, 1-line token-optimized lookups.

A **flow** is a path through the skills. Most paths run along the **main kickoff flow**, and two **on-ramps** merge onto it. Everything else is standalone or a specialized domain skill.

## The main flow: idea → ship (`kickoff`)

The route most work travels. You have an idea and want it built.

1. **`Phase 0 (Project Scan)`** — Read `.agents/SUMMARY_INDEX.md` and repo landscape.
2. **`Phase 1 (User Vision Intake)`** — Open-ended intake + `grill-me` constructive critique.
3. **`Phase 2 (Big Decision Grilling)`** — Run `grilling` to resolve foundational choices (Why, Stack, Architecture, Constraints) **before** `wayfinder`.
4. **`Phase 3 (Wayfinder)`** — `wayfinder` maps remaining unknowns and decision tickets.
5. **`Phase 4 (Code Review)`** — `code-review` audits existing code against standards if applicable.
6. **`Phase 5 (Deep Grilling)`** — `grill-with-docs` / `grilling` Q&A on edge cases, populating `CONTEXT.md` & ADRs.
7. **`Phase 6 (Spec Synthesis)`** — `to-spec` writes `.agents/ACTIVE_SPEC.md`.
8. **`Phase 7 (Ticket Breakdown)`** — `to-tickets` partitions spec into tracer-bullet tickets.
9. **`Phase 8 (Incremental TDD Implementation)`** — `implement` + `tdd` per ticket, user verification, optional git commit per ticket, `code-review` in clean context window.

## Persistent Prompt-Level Grilling Protocol (`grill-me`)

On EVERY user prompt requesting changes (even outside `kickoff`), the agent must:
- Check for ambiguities and conflicts against `.agents/ACTIVE_SPEC.md` and `.agents/CONTEXT.md`.
- Evaluate alternative implementation methods/logic.
- Check `.agents/SKILLS_INDEX.md` for applicable skills (e.g. `frontend-design`, `vercel-react-best-practices`, `web-design-guidelines`, `webapp-testing`, `api-design`).
- Reach shared understanding before generating code.

## On-ramps

- **Bugs and requests piling up** → `triage`.
- **Something broken / slow / flaking** → `diagnosing-bugs`.
- **Huge, foggy greenfield effort** → `wayfinder`.

## Standalone & Specialized Skills

- **Frontend & UI Polish:** `frontend-design`, `web-design-guidelines`, `vercel-react-best-practices`, `webapp-testing`.
- **Backend & Module Design:** `api-design`, `codebase-design`, `design-an-interface`.
- **Quick Verification:** `prototype` (throwaway code to test an idea fast).
- **Background Research:** `research` (investigate docs/APIs with cited markdown report).
- **Session Bridge:** `handoff` (compact context into markdown file to fork into a fresh window).
