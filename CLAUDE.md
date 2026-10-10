# IncarnaMind

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `junruxiong/IncarnaMind`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Design System
Read DESIGN.md before visual or UI work: it defines the fonts, colors, spacing, and
aesthetic direction. Ask the user before departing from it. When reviewing or QA-ing
UI, flag code that doesn't match DESIGN.md.

## Interaction rules
Read `docs/agents/interaction.md` before building or changing any UI. The bar: easy to
use and easy to pick up, not fewer features. Edit in place, drag with a non-drag
alternative, never replace the Mind the User is in, Undo instead of confirmations,
one right-click menu order, one selection model, keyboard and ⌘K for everything,
feedback within 100 ms, layouts that hold at every width in English and Chinese. A UI
pull request ticks the template's interaction checklist and shows before and after
screenshots. When briefing another agent on UI work, pass these rules on.
