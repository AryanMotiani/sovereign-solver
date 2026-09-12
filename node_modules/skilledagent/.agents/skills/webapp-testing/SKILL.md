---
name: webapp-testing
description: End-to-end web app testing, Playwright integration, visual regression testing, and assertion patterns.
disable-model-invocation: true
---

# Web Application Testing Skill

Use this skill when setting up, writing, or executing end-to-end (E2E) and integration tests for web applications.

## 1. Testing Methodology
- **User-Centric Locators:** Locate UI elements using role (`getByRole`), text (`getByText`), or label (`getByLabel`) rather than brittle CSS selectors.
- **Visual Assertions:** Verify page state visually by capturing screenshots or checking key element visibility before asserting logic.
- **Isolated State:** Ensure each test scenario starts with a clean state (seeded test DB, cleared browser storage, isolated session context).

## 2. Playwright / Cypress Execution Pattern
```typescript
import { test, expect } from '@playwright/test';

test('user can complete alignment workflow', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.getByRole('button', { name: /start/i }).click();
  await expect(page).toHaveURL(/\/dashboard/);
});
```
