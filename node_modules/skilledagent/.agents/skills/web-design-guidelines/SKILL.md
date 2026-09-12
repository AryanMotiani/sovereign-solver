---
name: web-design-guidelines
description: Audits UI code against 100+ rules covering accessibility (WCAG), responsive layouts, and visual hierarchy.
disable-model-invocation: true
---

# Web Design & Accessibility Guidelines

Use this skill when auditing UI code for usability, accessibility (WCAG compliance), and responsive rendering across devices.

## 1. Accessibility (a11y) Rules
- **Semantic HTML5:** Use `<main>`, `<nav>`, `<header>`, `<footer>`, `<section>`, and `<article>` tags instead of generic `<div>` containers.
- **Unique Element IDs:** Ensure interactive form elements, buttons, and sections carry unique `id` attributes.
- **Form Labeling:** Every input MUST have an associated `<label for="...">` or `aria-label`.
- **Keyboard Navigation:** All interactive elements must receive visible `:focus-visible` outline indicators and support `Tab` / `Enter` navigation.
- **Color Contrast:** Text must satisfy WCAG AA contrast ratio (4.5:1 for standard text, 3:1 for large text).

## 2. Responsive & Fluid Layouts
- Use CSS Grid and Flexbox for fluid layouts.
- Avoid hardcoded static pixel dimensions for main content containers; use `max-width`, `rem`, and percentage units.
