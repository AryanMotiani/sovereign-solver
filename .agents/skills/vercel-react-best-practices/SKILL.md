---
name: vercel-react-best-practices
description: Rules for React and Next.js performance, server components, bundle size optimization, and state management.
disable-model-invocation: true
---

# Vercel React & Next.js Best Practices

Use this skill when developing React, Next.js, or TypeScript web applications.

## 1. Core Engineering Rules
- **Server Components First:** Default to React Server Components (RSC) unless dynamic client interactivity or hooks (`useState`, `useEffect`, event handlers) are explicitly required.
- **Minimize Client Bundle Size:** Keep `"use client"` directives at the leaves of your component tree to prevent bundling heavy server code on the client.
- **Eliminate Hydration Mismatches:** Ensure non-deterministic values (dates, random numbers, browser localStorage) are initialized inside `useEffect` or suppressed properly.
- **Prevent Request Waterfalls:** Fetch data in parallel (`Promise.all`) or lift fetch calls high up in the component tree to avoid sequential component-level fetching.

## 2. State & Effect Optimization
- Do not use `useEffect` for state transformation that can be calculated during render.
- Pass stable callback references using `useCallback` when passing handlers to memoized child components.
- Use explicit key props for dynamic list rendering.
