# ⚙️ Agentic Skeleton: Operations & Command Guide

This directory contains all required AI skills, rules, and configuration. The following guide dictates how to use this template in your project.

> [!NOTE]
> This folder is designed to bootstrap AI-assisted development. All files live inside `.agents/` so you can easily remove the template from your project at any time.

---

## ⚖️ The Golden Laws (What You Must Enforce)

As the Orchestrator, you must constantly hold the AI accountable to these standards. If it violates them, halt execution.

1. **The Pedagogical Mandate:** Your goal is mastery over deep backend systems and AI/ML architectures. If the agent outputs complex logic (e.g., tensor operations, advanced concurrency, or strict Pydantic validation), force it to explain the _why_ before you accept the code.
2. **Anti-Vibe Coding:** Code is never written without a blueprint. The Red-Green-Refactor loop (`/tdd`) is absolute.
3. **Data Rigor:** Never accept flat database schemas. Enforce BCNF normalization and robust concurrency management (avoiding reader-writer starvation).
4. **Context Authority:** The agent is forbidden from silently changing its own rules. All updates to `.agents/CONTEXT.md` or the skills must be transparent and verified by you.

---

## 🚀 Scaffolding & Setup Guide

This operations guide is copied into your project's `.agents/` directory.

### What Gets Copied

The template scaffold includes:

- **AI Skills:** Pre-configured capabilities (located in `.agents/skills/`) for standard developer tasks.
- **Workflows:** Orchestration pipelines (located in `.agents/workflows/`) that chain skills together.
- **CONTEXT.md:** Global rules, ubiquitous language, and system architecture definitions (located in `.agents/CONTEXT.md`).
- **AGENTS.MD:** Workspace mapping and skill routing tables (located in `.agents/AGENTS.MD`).
- **skills-lock.json:** Secure lockfile specifying installed capabilities (located in `.agents/skills-lock.json`).

> [!NOTE]
> The scaffold ONLY adds AI workflow and configuration files inside `.agents/`. Your project source code remains untouched, and the entire template can be cleaned up cleanly by deleting this folder.

---

## 🛤️ Choose Your Workflow

### Workflow A — New Project

If starting a brand new project in an empty directory:

1. **Initialize Directory & Git:**
   ```bash
   mkdir MyProject
   cd MyProject
   git init
   ```
2. **Scaffold the Workspace:**
   ```bash
   npx -y giget@latest github:AryanMotiani/SkilledAgent/.agents .agents
   ```
   _(Note: Replace `AryanMotiani/SkilledAgent` with your own repository path if you have forked or customized the template)._

---

### Workflow B — Existing Repository

If injecting the template into your primary project repository:

1. **Navigate to project directory:**
   ```bash
   cd your-existing-project-folder
   ```
2. **Download the Template Folder:**
   ```bash
   npx -y giget@latest github:AryanMotiani/SkilledAgent/.agents .agents
   ```
   _(Note: Replace `AryanMotiani/SkilledAgent` with your own repository path if you have forked or customized the template. Since everything is isolated within `.agents`, it will not conflict with or overwrite any root files, like your project's `README.md` or `.gitignore`!)_

---

## 🔍 Verification & First Steps

### 1. Verify Scaffolded Files

Confirm that your target project directory now contains the following:

```text
.agents/
  ├── skills/
  ├── workflows/
  ├── CONTEXT.md
  ├── AGENTS.MD
  ├── AGENTS_README.md
  └── skills-lock.json
```

_(Note: The skills are copied directly into the project and should be available to your AI agent. If they are not detected immediately, reload the workspace or restart the AI session.)_

### 2. Initiate the Project Kickoff (`kickoff`)

Once the folder has been scaffolded, open it in your IDE (e.g. VS Code, Cursor), and start the `kickoff` workflow. Paste this prompt into the AI agent chat:

> "I have a new project idea: `[INSERT YOUR PROJECT IDEA/CONCEPT]`. Run the `kickoff` workflow."

This launches the full orchestrated pipeline:

- **Phase 0 — Project Scan:** Read and digest the project files to understand what's already here.
- **Phase 1 — User Vision Intake:** Ask you for your idea, goals, priorities, and every detail, acting as a constructive critic to identify plan cons, risks, and alternatives.
- **Phase 2 — Big Decision Grilling:** Resolve foundational project decisions (why, stack, architecture, deployment).
- **Phase 3 — Wayfinder:** Chart the decision map, explore the problem space, surface remaining unknowns.
- **Phase 4 — Code Review (if existing code):** Analyze existing codebase against standards.
- **Phase 5 — Deep Grilling:** Relentless Q&A on architecture and edge cases — no holes, no ambiguity.
- **Phase 6 — Spec Synthesis:** Generate the formal spec via `/to-spec`.
- **Phase 7 — Ticket Breakdown:** Break the spec into tracer-bullet vertical-slice tickets via `/to-tickets`.
- **Phase 8 — Implementation:** Execute tickets under TDD via `/implement`.
- **Phase 9 — Post-Implementation Review:** Final `/code-review` pass.
- **Phase 10 — Suggestions:** Ideas, problems, and improvements for the project.
- **Phase 11 — Skill Discovery:** Explore other available skills.

### 3. Implement Tasks under strict TDD

Once the tickets are generated, pick up the first ticket and trigger the implementation loop:

> "Let's implement ticket #1. Use `/implement` to execute it under TDD."

---

## 🗑️ How to Delete the Template

If you ever want to completely remove this scaffolding, its custom skills, and all AI rules from your repository, run the command for your shell:

**Linux / macOS (Bash/Zsh):**

```bash
rm -rf .agents
```

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse -Force .agents
```

**Windows (CMD):**

```cmd
rmdir /s /q .agents
```

This leaves zero leftover files in your repository.
