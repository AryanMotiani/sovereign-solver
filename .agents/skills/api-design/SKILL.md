---
name: api-design
description: Designing RESTful and GraphQL API contracts, module interfaces, error schemas, and input validation.
disable-model-invocation: true
---

# API Design & Module Interface Skill

Use this skill when designing backend microservice APIs, REST/GraphQL schemas, or internal TypeScript module interfaces.

## 1. RESTful & GraphQL Standards
- **Resource-Oriented Endpoints:** Use standard HTTP verbs (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`) with plural resource nouns (`/api/v1/users`, `/api/v1/tickets`).
- **Standardized Error Payload Schema:**
  ```json
  {
    "error": {
      "code": "INVALID_INPUT",
      "message": "Field 'email' must be a valid email address.",
      "details": [{ "field": "email", "issue": "format" }]
    }
  }
  ```
- **Strict Input Validation:** Validate all request payloads at the boundary using schema validators (Zod, TypeBox, Joi) before passing data to domain logic.

## 2. Deep Module Interface Design
- **Small Interface, Deep Functionality:** Expose a minimal, intuitive surface area while hiding complex internal implementation details behind clean module boundaries.
- **Explicit Parameter Objects:** Avoid long positional argument lists; accept structured, typed options objects.
