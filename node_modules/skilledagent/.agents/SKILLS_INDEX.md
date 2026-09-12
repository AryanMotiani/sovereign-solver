# SkilledAgent Skills Index (Token-Optimized Lookup)

> **Agent Instruction:** Read this lightweight index FIRST to match skills to user prompts. Do NOT read full `SKILL.md` files unless a skill is actively triggered.

| Skill Name | Triggers / Keywords | 1-Line Summary | Path |
|---|---|---|---|
| `ask-matt` | "what skill to use", "how to build", "router" | High-level router explaining which skill or flow fits your situation. | `.agents/skills/ask-matt/SKILL.md` |
| `api-design` | "api design", "rest api", "graphql schema", "contracts" | Designing robust REST/GraphQL contracts, error models, and module boundaries. | `.agents/skills/api-design/SKILL.md` |
| `claude-handoff` | "claude handoff", "export session" | Handoff session context specifically formatted for Claude. | `.agents/skills/claude-handoff/SKILL.md` |
| `code-review` | "code review", "review branch", "PR review", "spec review" | Two-axis code review (Standards + Spec compliance) using parallel evaluation. | `.agents/skills/code-review/SKILL.md` |
| `codebase-design` | "codebase design", "deep modules", "interfaces", "seams" | Shared vocabulary for designing deep modules with small interfaces. | `.agents/skills/codebase-design/SKILL.md` |
| `design-an-interface` | "design interface", "API options", "compare module shapes" | Generates multiple radically different interface designs for comparison. | `.agents/skills/design-an-interface/SKILL.md` |
| `diagnosing-bugs` | "diagnose bug", "debug", "throws error", "slow", "flake" | Scientific diagnosis loop for hard bugs and performance regressions. | `.agents/skills/diagnosing-bugs/SKILL.md` |
| `domain-modeling` | "domain model", "ubiquitous language", "glossary", "ADR" | Sharpen domain language, resolve overloaded terms, and record ADRs. | `.agents/skills/domain-modeling/SKILL.md` |
| `edit-article` | "edit article", "proofread prose", "polish writing" | Structural and stylistic editor for technical prose and articles. | `.agents/skills/edit-article/SKILL.md` |
| `frontend-design` | "frontend design", "ui design", "css styling", "component visual" | Guidance on visual aesthetics, HSL color palettes, animations, and UI polish. | `.agents/skills/frontend-design/SKILL.md` |
| `git-guardrails-claude-code` | "git safety", "block push", "prevent destructive git" | Hooks to block dangerous git operations (push, reset --hard, clean -fd). | `.agents/skills/git-guardrails-claude-code/SKILL.md` |
| `grill-me` | "grill me", "interview plan", "stateless Q&A", "clarify prompt" | Relentless interactive interview to stress-test a plan without writing context files. | `.agents/skills/grill-me/SKILL.md` |
| `grill-with-docs` | "grill with docs", "stateful grilling", "interview codebase" | Stateful interview that sharpens ideas, saving decisions into `CONTEXT.md` & ADRs. | `.agents/skills/grill-with-docs/SKILL.md` |
| `grilling` | "grilling", "q&a decision", "interview developer" | Primitive relentless Q&A loop used by `grill-me` and `grill-with-docs`. | `.agents/skills/grilling/SKILL.md` |
| `handoff` | "handoff", "fresh session", "compact context" | Compacts session context into a file to fork work into a fresh window. | `.agents/skills/handoff/SKILL.md` |
| `implement` | "implement ticket", "build task", "execute feature" | Implementation engine running TDD internally, ending with a code review. | `.agents/skills/implement/SKILL.md` |
| `improve-codebase-architecture` | "messy architecture", "deepening opportunities" | Scans module debt, surfaces deepening opportunities, generates HTML report. | `.agents/skills/improve-codebase-architecture/SKILL.md` |
| `loop-me` | "loop me", "automate workflow", "repeating task" | Generates specification for automated background loops. | `.agents/skills/loop-me/SKILL.md` |
| `migrate-to-shoehorn` | "shoehorn", "replace type assertion", "test data mock" | Migrates test files from `as` type assertions to shoehorn helper functions. | `.agents/skills/migrate-to-shoehorn/SKILL.md` |
| `obsidian-vault` | "obsidian vault", "manage notes", "wikilinks" | Search, create, and organize notes inside Obsidian vault structure. | `.agents/skills/obsidian-vault/SKILL.md` |
| `prototype` | "prototype", "throwaway code", "verify design idea" | Builds throwaway code to answer a design or UI question before building. | `.agents/skills/prototype/SKILL.md` |
| `qa` | "qa session", "report bugs", "file issue" | Conversational QA session that explores codebase and files GitHub issues. | `.agents/skills/qa/SKILL.md` |
| `request-refactor-plan` | "refactor plan", "safe refactor", "incremental commits" | Plans refactor into tiny safe commits via user interview, filed as issue. | `.agents/skills/request-refactor-plan/SKILL.md` |
| `research` | "research topic", "look up docs", "investigate api" | Background primary-source research leaving cited markdown file in repo. | `.agents/skills/research/SKILL.md` |
| `resolving-merge-conflicts` | "merge conflict", "rebase conflict", "git conflict" | Step-by-step resolution of git merge or rebase conflicts. | `.agents/skills/resolving-merge-conflicts/SKILL.md` |
| `scaffold-exercises` | "scaffold exercise", "problem stubs", "course exercises" | Creates exercise directory structures with problems, solutions, and tests. | `.agents/skills/scaffold-exercises/SKILL.md` |
| `setup-matt-pocock-skills` | "setup skills", "configure tracker", "setup matt pocock" | Configures issue tracker, triage labels, and domain doc layout. | `.agents/skills/setup-matt-pocock-skills/SKILL.md` |
| `setup-pre-commit` | "setup pre-commit", "husky", "lint-staged" | Installs Husky and lint-staged for commit-time linting, formatting, testing. | `.agents/skills/setup-pre-commit/SKILL.md` |
| `setup-ts-deep-modules` | "setup ts deep modules", "typescript encapsulation" | Sets up TypeScript strict boundary checks for deep module enforcement. | `.agents/skills/setup-ts-deep-modules/SKILL.md` |
| `tdd` | "tdd", "red-green-refactor", "unit test first" | Red-green-refactor TDD loop enforcing failing tests before implementation. | `.agents/skills/tdd/SKILL.md` |
| `teach` | "teach me", "explain concept", "learn codebase" | Multi-session pedagogical instruction using repo as interactive workspace. | `.agents/skills/teach/SKILL.md` |
| `to-spec` | "to spec", "write spec", "technical specification" | Synthesizes project decisions into a formal technical specification. | `.agents/skills/to-spec/SKILL.md` |
| `to-tickets` | "to tickets", "task breakdown", "tracer bullet tickets" | Splits spec into vertical-slice tickets with declared blocking dependencies. | `.agents/skills/to-tickets/SKILL.md` |
| `triage` | "triage issues", "evaluate bug reports", "incoming issues" | State-machine triage role manager for issue trackers and external PRs. | `.agents/skills/triage/SKILL.md` |
| `ubiquitous-language` | "ubiquitous language", "domain vocabulary" | Extracts and maintains domain glossary terms across code and docs. | `.agents/skills/ubiquitous-language/SKILL.md` |
| `vercel-react-best-practices` | "react best practices", "nextjs optimization", "bundle size" | 40+ rules for React/Next.js performance, hydration, and bundle minimization. | `.agents/skills/vercel-react-best-practices/SKILL.md` |
| `wayfinder` | "wayfinder", "chart map", "unknowns", "explore project" | Charts shared map of investigation tickets for foggy, large greenfield efforts. | `.agents/skills/wayfinder/SKILL.md` |
| `web-design-guidelines` | "web design guidelines", "accessibility", "ui rules" | Audits UI code against 100+ rules covering WCAG accessibility and UX. | `.agents/skills/web-design-guidelines/SKILL.md` |
| `webapp-testing` | "webapp testing", "playwright testing", "e2e test" | Visual and functional end-to-end web application testing using Playwright. | `.agents/skills/webapp-testing/SKILL.md` |
| `wizard` | "wizard", "cli setup guide", "interactive wizard" | Generates interactive CLI setup guides for complex repo tooling. | `.agents/skills/wizard/SKILL.md` |
| `writing-beats` | "writing beats", "outline structure", "prose beats" | Sequences raw material into coherent outline beats. | `.agents/skills/writing-beats/SKILL.md` |
| `writing-fragments` | "writing fragments", "raw ideas", "brainstorm writing" | Collects and organizes raw material fragments before structuring. | `.agents/skills/writing-fragments/SKILL.md` |
| `writing-great-skills` | "write skill", "create skill", "skill guidelines" | Authoritative guide and standard patterns for writing new agent skills. | `.agents/skills/writing-great-skills/SKILL.md` |
| `writing-shape` | "writing shape", "draft prose", "polish document" | Shapes structured beats into polished technical prose. | `.agents/skills/writing-shape/SKILL.md` |
