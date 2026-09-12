# Universal Engineering Context & Behavioral Guide

## 1. System Guardrails & Execution Philosophy
- **Persistent SkilledAgent Policy:** SkilledAgent is ALWAYS ACTIVE. Every user prompt requesting code modifications must be evaluated via prompt-level grilling (`grill-me`), checking for ambiguities, spec conflicts in `.agents/ACTIVE_SPEC.md`, implementation logic choices, and skill matches in `.agents/SKILLS_INDEX.md`.
- **Token Reduction & Lazy Loading Protocol:** 
  1. *Lazy Skill Loading:* Match skills via `.agents/SKILLS_INDEX.md` first; load full `SKILL.md` files only on active trigger.
  2. *Summary Caching:* Consult `.agents/SUMMARY_INDEX.md` instead of re-scanning entire directories on every prompt.
  3. *Scoped Line Viewing:* Read files in narrow line slices (20-100 lines max).
  4. *Concise Outputs:* Produce direct, token-dense answers without repeating context history.
- **Strict 12k Limit:** Observe a 12,000 line / ~120k token ceiling per markdown file to protect context reasoning.
- **Strict TDD:** Never write production application implementations without an accompanying failing test (Strict Red-Green-Refactor loop).
- **Pedagogical Requirement:** Explain architectural patterns (schema normalization, concurrency locks, interface depth) when presenting solutions. Elevate the developer's understanding.
- **Workflow Obedience:** Follow the sequential orchestration defined in the `kickoff` workflow (Scan → Vision Intake → Big Decisions → Wayfinder → Deep Grilling → Spec → Tickets → Incremental TDD Implementation → Review).

## 2. Architectural Discipline & Alignment
- **ADR Mandate:** Any major deviation from the initial spec requires a new Architecture Decision Record (ADR) before implementation.
- **Alignment Protocol:** Prioritize data isolation, interface depth, and concurrency safety.
- **Dynamic Stack Injection:** Project specifications populate during the `kickoff` alignment phase and persist in `.agents/ACTIVE_SPEC.md`.

## 3. Technical Standards (To Be Populated During Project Kickoff)
- *(Agent: Actively populate this section during the kickoff workflow interview or spec onboarding with the target languages, frameworks, testing libraries, and architectural conventions selected for the project. For example: Node.js/TypeScript, PostgreSQL, NestJS, Jest, etc. Enforce these strictly as the absolute engineering baseline.)*

## 4. Ubiquitous Language (Domain Dictionary)
- *(Agent: Actively populate this section during the kickoff workflow interview with domain-specific terminology, business logic definitions, and common words to ensure ubiquitous language across all agentic skills).*