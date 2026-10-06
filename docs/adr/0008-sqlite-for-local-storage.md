# SQLite is the production database for the desktop app

The desktop app (ADR-0004) stores everything except Mind content and Document files in a single SQLite database in the User's data folder. That includes Document metadata, Passages and their embeddings, settings, Connectors and approvals. SQLite is not a stand-in for a "real" database here. It is the standard production choice for local apps (browsers, phones and Lightroom all rely on it). It needs no database server for the User to install, handles one person's library easily, and backing up is just copying the data folder. Its known weakness is many concurrent writers, which a single-user desktop app never has.

## Considered options

- **PGlite** (Postgres running inside the app): would match a future Postgres-backed hosted version, but it is heavier and less mature.
- **A Postgres server**: rejected, because it would require Users to install and run a database.

## Consequences

The storage layer in the core is written so it can be swapped. The hosted version may use Postgres or one SQLite database per User, decided when it is built.
