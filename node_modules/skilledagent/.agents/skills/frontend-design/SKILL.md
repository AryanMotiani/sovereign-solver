---
name: frontend-design
description: Guidance on visual aesthetics, modern CSS, color systems, glassmorphism, animations, and component styling.
disable-model-invocation: true
---

# Frontend Design & Aesthetics Skill

Use this skill when building or styling web applications, components, or UI interfaces.

## 1. Aesthetic Excellence Guidelines
- **Modern Typography:** Use modern sans-serif fonts (Inter, Roboto, Outfit, Plus Jakarta Sans) rather than browser defaults.
- **Curated Color Palettes:** Avoid raw primary colors (`#ff0000`, `#0000ff`). Use HSL-tailored scales, subtle dark modes, glassmorphism (`backdrop-filter: blur()`), and sleek border gradients.
- **Dynamic Micro-Animations:** Add CSS transitions (`transition: all 0.2s ease-in-out`), hover scale effects (`transform: translateY(-2px)`), and active states to make the interface feel responsive and alive.
- **Layout Depth & Spacing:** Use consistent spacing scales (rem/px tokens), subtle box-shadows, dynamic card borders, and clear visual hierarchy.

## 2. No Placeholders Policy
- Never use placeholder boxes or broken asset links.
- Generate image assets or render clean CSS SVG visuals.

## 3. CSS Tokens Pattern
Define standard design tokens at the root level:
```css
:root {
  --bg-primary: #0f172a;
  --bg-card: rgba(30, 41, 59, 0.7);
  --border-subtle: rgba(255, 255, 255, 0.1);
  --accent-primary: #6366f1;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
}
```
