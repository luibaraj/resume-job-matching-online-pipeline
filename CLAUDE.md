# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Principles

**Philosophy: Do the simplest thing that works.**

### Simplicity & Conciseness

- Write the minimum code needed to solve the problem correctly — no more
- Delete code that isn't doing meaningful work; dead code is a liability
- Prefer a clear 10-line function over an abstracted 50-line one
- No speculative abstractions: don't build for hypothetical future requirements
- Create/Edit the least number of files necessary for each task

### Logging

Log only where execution is hard to trace: proxy failures, unexpected branches, and operation outcomes. Avoid logging in straightforward code paths.

- `DEBUG` — tricky request-level details
- `INFO` — top-level operation start/finish
- `WARNING` — recoverable failures (retries, rotations)
- `ERROR` — unrecoverable failures

### What to Avoid

- Commented-out code (delete it; git has history)
- Wrapper functions that only call one other function with no added logic
- Generic exception handling that silently swallows errors (`except Exception: pass`)
