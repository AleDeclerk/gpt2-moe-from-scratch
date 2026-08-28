# web

The website of the course. It renders the chapters and the measured progress.

The application reads three things from the repository, one level up:

| Source | Use |
|---|---|
| `chapters/manifest.json` | The list of the 14 chapters, and the order |
| `chapters/*/README.md` | The theory of each chapter |
| `progress.json` | The result of each test, from `scripts/sync_progress.py` |

The theory has one home only, which is the README of the chapter. The site
does not hold a copy.

## Local

```bash
npm install
npm run dev
```

The build is static. To see a change of the progress, run
`uv run python scripts/sync_progress.py` in the root of the repository first.

## Deploy

Vercel builds this directory from the root of the repository, with the
commands in `vercel.json`. A push to `main` starts a deployment.
