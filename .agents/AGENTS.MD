# Repository Architecture & Rules (`AGENTS.MD`)

## a. How this Repo Works & SkilledAgent Core Behavior
- **Persistent Agent Enforcement:** SkilledAgent is ALWAYS active. Once loaded, it intercepts EVERY user prompt requesting changes—whether in `kickoff` or in a standalone prompt—to clear ambiguities, verify spec alignment, present logic choices, check available skills, and reach a shared understanding before writing code.
- **Directory Layout:**
  - `src/`: Core application modules.
  - `tests/`: Automated test suites accompanying every module.
  - `.agents/`: Custom behaviors, index files, skills, and workflows.
  - **Agent Instruction:** You MUST scan `.agents/SKILLS_INDEX.md` and `.agents/SUMMARY_INDEX.md` first to check available skills and project state before reading full source files or full `SKILL.md` documents.

## b. Persistent Prompt-Level Grilling Protocol (`grill-me`)
- **Trigger:** EVERY user prompt that requests changes or new functionality.
- **Execution:**
  1. **Check Ambiguities & Conflicts:** Evaluate the prompt against `.agents/ACTIVE_SPEC.md` and `.agents/CONTEXT.md`. Highlight any missing details or conflicting requirements.
  2. **Option Evaluation:** If more than one implementation logic/method exists, present the options clearly with recommendations.
  3. **Token-Optimized Skill Check:** Consult `.agents/SKILLS_INDEX.md` (without parsing heavy SKILL files) to check if existing skills can assist. Ask the user if available extra skills should be used.
  4. **Shared Understanding:** Confirm alignment with the user before generating code or modifying files.

## c. Skill Routing & Router Table
- Primary Router: `ask-matt` (Guides selection of workflows, on-ramps, and standalone skills).
- User-Initiated Pipeline: `kickoff` workflow (Guides project scan, vision intake, big decision grilling, wayfinder mapping, deep Q&A, spec synthesis, ticket breakdown, and incremental TDD implementation).
- Model-Initiated Utility Frameworks:
  - Use `to-spec` for functional system design synthesis (`.agents/ACTIVE_SPEC.md`).
  - Use `to-tickets` for backlog distribution into tracer-bullet vertical slices.
  - Use `implement` for execution (runs `tdd` and `code-review` internally).
  - Use `wayfinder` for charting maps of decision tickets for greenfield or foggy tasks.
  - Use `grilling`, `grill-me`, or `grill-with-docs` for interactive Q&A sessions.
  - Use `prototype` for throwaway code to test design/UI ideas.
  - Use `triage` for issue tracker evaluation and agent brief creation.

## d. Knowledge & Token Architecture
- Core engineering truths and design patterns are maintained continuously in `.agents/CONTEXT.md`.
- Active feature specifications and backlog state live in `.agents/ACTIVE_SPEC.md` and `.agents/SUMMARY_INDEX.md`.
- **Token Guardrail:** Adhere to a 12,000 line / ~120k token limit per document. Always read summary indices first.

## e. Project Lifecycle (`kickoff`)
- **Pipeline Order:**
  1. **Project Scan** (Read landscape, config, dependencies, structure).
  2. **User Vision Intake** (Ask user for idea, goals, priorities, tech/non-tech details + constructive critique).
  3. **Big Decision Grilling** (`grilling` — resolve foundational decisions: why built, stack, architecture, deployment, constraints BEFORE wayfinder).
  4. **Wayfinder** (`wayfinder` — chart decision map & ticket fanout).
  5. **Deep Interview & Grilling** (`grilling` or `grill-with-docs` — detailed Q&A on edge cases, domain dictionary).
  6. **Spec Synthesis** (`to-spec` — generate formal spec in `ACTIVE_SPEC.md`).
  7. **Task Deconstruction** (`to-tickets` — vertical slice tickets with blocking dependencies).
  8. **Incremental Implementation Loop:**
     - Implement tickets **one by one** using `implement` / `tdd`.
     - Request user verification after each ticket.
     - Offer to commit after each change/ticket (`git commit`).
     - Run `code-review` in a clean context window (ask user to switch model / clear context if needed).

## f. Working Rules
- Hard Rule: SkilledAgent remains active across ALL turns and new conversations until explicitly disabled by user.
- Hard Rule: Never skip writing a failing test before building application logic or API routing structures.
- Hard Rule: Never mutate workflow or core instruction files autonomously without explicit user confirmation.

## g. Token Reduction & Optimization Protocol (Save 80%+ Context Tokens)
1. **Lazy Skill Loading (Index-First):** NEVER parse or view full `SKILL.md` documents during initial discovery or prompt grilling. Match user intent strictly against the 1-line triggers in `.agents/SKILLS_INDEX.md`. Load a specific `SKILL.md` file ONLY when that skill is explicitly triggered.
2. **Stateful Summary Caching:** Read `.agents/SUMMARY_INDEX.md` for project context instead of re-scanning directory trees or full source files on every turn. Update `SUMMARY_INDEX.md` inline when files or decisions change.
3. **Scoped File Reading:** Use narrow line ranges (`StartLine`/`EndLine`) when viewing files. Never load more than 100 lines at once unless strictly necessary.
4. **Concise Response Directives:** Keep agent responses crisp, bulleted, and direct. Do NOT re-summarize entire conversation history or echo full file contents back to the user.
5. **Phase Boundary Hygiene:** Recommend context compacting or clearing (`handoff`) between major phases (e.g. after `to-tickets` before `implement`, or between individual implementation tickets) to keep active memory in the model's high-efficiency zone (~10k-20k tokens).


