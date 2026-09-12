---
description: Full project pipeline — scan, vision intake, big decisions, wayfind, grill, spec, ticket, implement, review.
---

# kickoff — Orchestrated Project Pipeline

Chains engineering skills into an idea → production pipeline. Perform phases sequentially. Do not proceed without user confirmation.

> **Trigger:** `kickoff` (or `/kickoff`)
> **Supports:** New projects and existing codebase onboarding.

---

## 👋 Welcome & Plan Overview (Always Show First)

When the user runs `kickoff`, print this overview first:

> "Welcome to SkilledAgent! `kickoff` runs a structured 12-phase pipeline from idea to test-driven production code."

**Phase Plan:**
| # | Goal | Skill / Mechanism |
|---|---|---|
| 0 | Read and digest project files | Filesystem scan + `.agents/SUMMARY_INDEX.md` |
| 1 | Understand user's idea, goals, priorities | Open-ended intake + `grill-me` critique |
| 2 | Resolve foundational project decisions | `grilling` (Why, Stack, Architecture, Constraints) |
| 3 | Map remaining decisions & unknowns | `wayfinder` |
| 4 | Audit existing code (if any) | `code-review` |
| 5 | Architecture & edge-case Q&A | `grilling` + `grill-with-docs` |
| 6 | Write formal spec | `to-spec` → `.agents/ACTIVE_SPEC.md` |
| 7 | Break spec into tickets | `to-tickets` |
| 8 | Ticket-by-ticket TDD implementation | `implement` + `tdd` + user verify + git commit |
| 9 | Post-implementation audit | `code-review` (clean context window) |
| 10 | Final review & recommendations | Synthesis |
| 11 | Other skills discovery catalog | Tour (44 Skills) |

> "Before each phase, I will **announce the skill** and **why** I'm using it. I will always ask for confirmation before moving on."

---

## 🧭 Persistent Prompt Grilling & Skill Matching (Always Active)

For ANY user prompt (even outside `kickoff`), SkilledAgent automatically executes:
1. **Ambiguity & Spec Check:** Evaluates user prompts against `ACTIVE_SPEC.md` and `CONTEXT.md`.
2. **Implementation Options:** Surface alternative logic/methods if multiple choices exist.
3. **Token-Saving Index Check:** Consults `SKILLS_INDEX.md` to identify relevant skills (including top skills.sh integrations like `frontend-design`, `vercel-react-best-practices`, `web-design-guidelines`, `webapp-testing`, `api-design`).
4. **Shared Understanding:** Asks clarifying questions and confirms alignment before touching code.

---

## Phase 0: Project Scan — Read the Landscape

_No special skill — lightweight summary scan_

> 🔧 "I'm scanning your project directory — files, config, dependencies, summary indices — so I have full context before asking you anything."

1. Read `.agents/SUMMARY_INDEX.md` and scan `src/`, config files, README, docs, tests, `.agents/` config.
2. Summarize: detected languages/frameworks, folder structure, tests/CI, fresh vs existing codebase.
3. Ask: "Is this picture accurate? Anything I missed?"

**Exit criteria:** User confirms the scan is accurate.

---

## Phase 1: User Vision Intake — What Are You Building?

**Skills:** Open-ended intake + `grilling` (critique mode / `grill-me`)

> 🔧 "Now I need to hear your vision. This step ensures we are on the **same page** about the final project idea so that any holes in the plan are identified and filled before we build anything."

**Why this phase exists:** The goal is shared understanding. The user describes their vision freely, and the agent listens, then acts as a **constructive critic** — identifying possible cons, gaps, and risks in the plan and suggesting resolutions.

1. Ask the user:
   > "Tell me what you're trying to implement — your idea, how you want to build it, your priorities, and every technical and non-technical detail you can think of."
2. Listen fully. Do not interrupt or start breaking things down yet.
3. Summarize back what you heard and confirm understanding.
4. **Critique the plan:** Act as a constructive critic using `grilling`:
   - Identify possible cons, risks, or weak points in the user's approach
   - Surface assumptions that might not hold
   - Suggest alternative approaches or resolutions where relevant
   - Use `research` if a claim needs verification against primary sources
   - Use `codebase-design` or `api-design` if a proposed module structure has deep-module concerns
5. Present your critique and suggestions. Iterate until the user is satisfied.

**Exit criteria:** User and agent agree on the project idea and have addressed identified risks.

---

## Phase 2: Big Decision Grilling — Foundational Questions

**Skill:** `grilling`

> 🔧 "Before breaking this into parts or mapping with wayfinder, I need to resolve the **big foundational decisions**. One question at a time."

1. Run `grilling` on **project-level foundational decisions** (BEFORE running `wayfinder`).
2. Ask one at a time (skip any answered in Phase 1):

   **Purpose:** Why is this being built? Who are the users? Success metrics?

   **Stack:** Language(s)? Framework(s) and why? Database (SQL/NoSQL)? Testing framework?

   **Architecture:** Monolith/microservices/serverless? API-first or UI-first? Monorepo/multi-repo? State management?

   **Deployment:** Cloud/self-hosted/edge? CI/CD? Environments (dev/staging/prod)?

   **Constraints:** Budget? Timeline? Team size? #1 priority (speed, quality, scalability)?

   **Business:** Business model? Compliance (GDPR, HIPAA)? Auth model? Third-party integrations?

3. For each question, provide your recommended answer.
4. Record all decisions — populate `.agents/CONTEXT.md` and `.agents/ACTIVE_SPEC.md`.

**Exit criteria:** All foundational decisions resolved. User agrees the big picture is locked.

---

## Phase 3: Wayfinder — Chart the Map

**Skill:** `wayfinder`

> 🔧 "Big decisions are locked. Using **`wayfinder`** to chart remaining unknowns and map the project."

1. Chart decision map with `wayfinder`, using context from Phases 0-2.
2. Define destination (what does "done" look like?).
3. Fan out breadth-first across open decisions/unknowns.
4. Create tracker issue/tickets.

**Exit criteria:** User approves the map or destination is clear.

---

## Phase 4: Code Review — Existing Codebase Audit (if applicable)

**Skill:** `code-review` _(Skip if starting from scratch)_

> 🔧 "Using **`code-review`** to audit existing code against standards and spec."

1. Ask user for branch, commit, or tag to compare against.
2. Run `code-review` (Standards and Spec axes).
3. Present findings, feeding into the Grilling phase.

**Exit Criteria:** User acknowledges findings.

---

## Phase 5: Deep Grilling — Architecture & Edge-Case Q&A

**Skill:** `grilling` + `grill-with-docs`

> 🔧 "Using **`grilling`** to interview you on detailed architecture, edge cases, and datastores."

1. Run `grilling` (or `grill-with-docs` for ADRs & glossary).
2. Q&A on architecture, edge cases, concurrency, validation. One question at a time.
3. Don't ask for codebase-discoverable facts; ask only for decisions.
4. Refine `UBIQUITOUS_LANGUAGE.md` using `domain-modeling`.

**Exit Criteria:** Both agree: "We have reached shared understanding."

---

## Phase 6: Spec Synthesis

**Skill:** `to-spec`

> 🔧 "Using **`to-spec`** to compile decisions into formal spec `.agents/ACTIVE_SPEC.md`."

1. Run `to-spec`. Write to `.agents/ACTIVE_SPEC.md`.
2. Include: Problem, Solution, User Stories, Architecture, Testing, Out of Scope.
3. Present for user review and iterate.

**Exit Criteria:** User says "Spec approved."

---

## Phase 7: Ticket Breakdown

**Skill:** `to-tickets`

> 🔧 "Using **`to-tickets`** to partition the spec into vertical-slice tickets."

1. Split spec into end-to-end task tickets.
2. Ensure vertical slices (DB, logic, UI, and tests together).
3. Connect dependencies. Present to user.

**Exit Criteria:** User approves breakdown.

---

## Phase 8: Incremental Ticket Implementation Loop

**Skills:** `implement` + `tdd` + `code-review`

> 🔧 "Implementing tickets **one by one** using test-driven development."

For EACH ticket in order:
1. Pick next unblocked ticket.
2. Run `implement` using strict TDD (`tdd` loop: failing test first, then minimal implementation code, then refactor).
3. Verify tests and type checks pass.
4. **User Verification:** Ask user to test and verify the ticket.
5. **Git Commit Option:** Ask user if they want to commit this change (`git commit`).
6. **Context Cleaning & Code Review:** Run `code-review` in a clean context window (ask user to switch model / clear context if possible).

**Exit Criteria:** All tickets implemented, tested, verified, and passing.

---

## Phase 9: Post-Implementation Code Review

**Skill:** `code-review`

> 🔧 "Using **`code-review`** to audit all new code against rules and spec."

1. Run `code-review` comparing against pre-implementation state.
2. Fix any warnings/violations.

**Exit Criteria:** Clean review on Standards and Spec.

---

## Phase 10: Suggestions & Next Steps

_Synthesis phase_

1. Review the deliverable against `SUMMARY_INDEX.md`.
2. Report: issues/debt, performance ideas, UX tweaks, security hardening.
3. Loop back if acting on suggestions.

**Exit Criteria:** User is satisfied or triggers a new `kickoff` loop.

---

## Phase 11: Skill Discovery Catalog (44 Available Skills)

Present other skills and ask what interests the user:

| Category | Recommended Skills |
|---|---|
| **Frontend & UI** | `frontend-design`, `vercel-react-best-practices`, `web-design-guidelines`, `webapp-testing` |
| **API & Architecture** | `api-design`, `codebase-design`, `design-an-interface`, `improve-codebase-architecture` |
| **Testing & Quality** | `tdd`, `code-review`, `qa`, `diagnosing-bugs` |
| **Planning & Routing** | `ask-matt`, `wayfinder`, `grill-me`, `grill-with-docs`, `to-spec`, `to-tickets`, `triage` |
| **Productivity & Utilities** | `prototype`, `research`, `handoff`, `request-refactor-plan`, `setup-pre-commit` |

Ask: _"Would you like to try any of these skills?"_

**Exit criteria:** User finishes or picks a skill.
