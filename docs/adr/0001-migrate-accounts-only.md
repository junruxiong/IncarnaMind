---
status: amended by ADR-0004
---

# Migrate user accounts only, not their content

When we replace the Django backend, user accounts carry over to the new app, but their Minds, Blocks and uploaded Documents do not. Returning users sign in to an empty workspace. We chose this because migrating content would mean translating Django's data shapes and re-processing every stored file, and that cost isn't justified. Once the Django database is shut down, that content is gone, so this decision is not reversible after cutover.

## Amendment (ADR-0004)

Version 1 is a desktop app with no sign-in, so the account migration happens when the hosted version exists, not at the rewrite. Before the Django database is shut down, export its user table, including password hashes and which users signed in with Google, and keep the export for the hosted version.
