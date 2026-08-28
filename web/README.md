# web

El sitio del curso. Renderiza los capítulos y el progreso medido.

La aplicación lee tres cosas del repositorio, un nivel más arriba:

| Fuente | Para qué sirve |
|---|---|
| `chapters/manifest.json` | La lista de los 14 capítulos, y el orden |
| `chapters/*/README.md` | La teoría de cada capítulo |
| `progress.json` | El resultado de cada test, que sale de `scripts/sync_progress.py` |

La teoría vive en un solo lugar: el README del capítulo. El sitio no guarda
una copia.

## Local

```bash
npm install
npm run dev
```

El build es estático. Si querés ver un cambio del progreso, corré primero
`uv run python scripts/sync_progress.py` en la raíz del repositorio.

## Deploy

Vercel construye este directorio desde la raíz del repositorio, con los
comandos de `vercel.json`. Un push a `main` dispara un deployment.
