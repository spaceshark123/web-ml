# Web ML

Small self-hosted web application for uploading datasets, training classical and neural-network models, visualizing training progress, and inspecting experiments.

## Key pieces

- Backend entry: [backend/app.py](backend/app.py) — minimal runner that loads the app.
- HTTP endpoints and Socket.IO handlers: [backend/endpoints.py](backend/endpoints.py).
- ML wrapper and model helpers: [`ModelWrapper`](backend/ml/wrapper.py) ([backend/ml/wrapper.py](backend/ml/wrapper.py)).
- DB models: [backend/models.py](backend/models.py).
- Shared extensions (db/socket/login): [backend/extensions.py](backend/extensions.py).
- Frontend: [frontend/](frontend/) (Vite + React + TypeScript).
- Frontend login/register pages: [frontend/src/pages/login.tsx](frontend/src/pages/login.tsx), [frontend/src/pages/register.tsx](frontend/src/pages/register.tsx).
- Training UI components: [frontend/src/components/train-model-dialog.tsx](frontend/src/components/train-model-dialog.tsx), [frontend/src/components/training-visualizer.tsx](frontend/src/components/training-visualizer.tsx).

## Requirements

- Python 3.10+ (backend)
- Node 18+ / npm or pnpm (frontend)
- Packages listed in [requirements.txt](requirements.txt) and [frontend/package.json](frontend/package.json)

## Quickstart — Backend

1. Create and activate a virtual environment and install dependencies:

```bash
# bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Initialize / reset the SQLite DB (an example `instance/app.db` is present). To recreate, stop the app, remove `instance/app.db`, then restart.

3. Run the backend (includes Socket.IO):

```bash
# bash
python app.py
```

This will start the server on port 5000 by default. The Flask app and Socket.IO handlers are wired in [backend/endpoints.py](backend/endpoints.py) and extensions come from [backend/extensions.py](backend/extensions.py).

## Quickstart — Frontend

1. Install dependencies and run dev server:

```bash
# bash
cd frontend
npm install
npm run dev
```

2. The frontend dev server (Vite) proxies /api to the backend using [frontend/vite.config.ts](frontend/vite.config.ts).

## Running tests

- Frontend unit tests use Jest. From the `frontend` directory:

```bash
# bash
cd frontend
npm test
```

- There are no automated backend tests included by default.

## API highlights

- Dataset endpoints and upload are implemented in [backend/endpoints.py](backend/endpoints.py) (e.g., `/api/upload`, `/api/datasets`).
- Model train endpoint (non-MLP streaming/non-streaming) is at `/api/train/<model_id>` in [backend/endpoints.py](backend/endpoints.py). For MLP streaming training the Socket.IO events (`start_training`, `pause_training`, `resume_training`, `early_stop_training`) are handled in the same file.
- Model serialization and evaluation helper is [`ModelWrapper`](backend/ml/wrapper.py).

## Development notes

- The code stores models and metadata in SQLite at `instance/app.db`. Models are pickled into the DB via [`ModelWrapper.to_db_record`](backend/ml/wrapper.py).
- Frontend components expect the backend API at `http://localhost:5000` (see [frontend/vite.config.ts](frontend/vite.config.ts) proxy).
- CORS is configured in the backend to allow the dev frontend origin (see [backend/endpoints.py](backend/endpoints.py) and [backend/extensions.py](backend/extensions.py)).
- Large files are stored under `uploads/` (configured via `app.config['UPLOAD_FOLDER']` in [backend/endpoints.py](backend/endpoints.py)).

### Where to start when changing code

- Add or modify endpoints in [backend/endpoints.py](backend/endpoints.py) or refactor into blueprints (recommended).
- Update ML logic in [`ModelWrapper`](backend/ml/wrapper.py) if adding new model types or metrics.
- Edit UI behavior in the corresponding frontend component files under [frontend/src/components/](frontend/src/components/).

### Reset / maintenance

- To reset uploads and DB for a clean slate:
- Stop the backend.
- Remove `instance/app.db` and contents of `uploads/`.
- Restart backend to recreate (some initialization logic runs on boot in [backend/endpoints.py](backend/endpoints.py)).

### Help / debugging

- Backend logs are printed to stdout when running `python backend/app.py`.
- The frontend dev server prints proxy activity (see [frontend/vite.config.ts](frontend/vite.config.ts) configure hooks).
- If Socket.IO connections fail in the browser, ensure the backend is reachable and `socket.io-client` version matches server compatibility.

## Contributing

- Follow the existing code structure. Consider splitting large files (for example split [backend/endpoints.py](backend/endpoints.py) into blueprints) and reuse shared extensions in [backend/extensions.py](backend/extensions.py).
