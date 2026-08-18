# Platform CRISP Web

React + TypeScript client for the Platform CRISP FastAPI service.

## Development

```bash
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`.

## Production

```bash
npm run build
cd ../..
python main.py web
```

FastAPI serves the generated `dist/` directory.
