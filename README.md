# pc_agent

Personal browser memory + lightweight action agent, powered by Gemini.

A research prototype that:

- **Captures** what you read and type in your browser (passively, with sensible defaults).
- **Remembers** it as a queryable personal memory you can ask natural-language questions about.
- **Acts** on your behalf for "go check X" tasks via Gemini function-calling and structured DOM extraction (no pixel-clicking).

Full quick-start, configuration, and privacy notes are added in a follow-up commit.

## Repo conventions

This repo uses a **per-repo git identity**. Run once after cloning:

```bash
./scripts/setup-git-identity.sh
```

This scopes `user.name` / `user.email` to this repo only (never touches your global config) and activates the in-tree pre-commit hook that hard-blocks commits with the wrong identity.

## License

MIT — see [LICENSE](LICENSE).
