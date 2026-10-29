# Joi's frontend

The chat screen uses **Vite, React, TypeScript, and Tailwind CSS**. The Python
server runs our own neural models and serves Vite's production build on the
same address: **http://127.0.0.1:7860**.

## Set up and open

Use Node.js 22.12+ with npm (this checkout was built with Node.js 24), plus the
Python environment and trained checkpoints from the [development guide](development.md).

```bash
npm --prefix frontend ci
npm --prefix frontend run build
uv run --offline python -m joi.serve
```

The first two commands install the locked frontend dependencies and build the
screen. Rebuild after changing the frontend. `frontend/dist/` and `node_modules/`
are generated locally and stay out of Git.

## Work on the screen

Keep the Python server running, then use a second terminal:

```bash
npm --prefix frontend run dev
```

Vite opens its development server at **http://127.0.0.1:5173** and forwards
`/api` requests to the Python server at port 7860. Changes refresh immediately.
The proxy sets the local API's expected origin; external origins remain rejected.

## Check changes

```bash
npm --prefix frontend test
npm --prefix frontend run build
uv run --offline python -m unittest discover -s tests
```

The frontend tests cover existing saved chats, complete context windows, duplicate
sends, reset during a pending reply, and keeping the draft after failed requests.
Backend tests cover built assets, missing builds, traversal protection, and chat
requests. Browser checks cover layout, keyboard input, the About dialog, real
local replies, and reload persistence.

## Where things live

- `frontend/src/App.tsx`: the fullscreen layout, composer, and connection status.
- `frontend/src/components/`: welcome, portrait, conversation, and About views.
- `frontend/src/lib/`: typed chat requests, saved history, and conversation state.
- `frontend/src/tailwind.css`: Tailwind's entry and font theme.
- `frontend/index.html`: Vite's minimal startup document; the screen lives in React.
- `frontend/vite.config.ts`: React, Tailwind, and the local development proxy.
- `joi/serve.py`: the local inference API and production asset server.

The old `web/` implementation is removed. Existing browser history keeps its
`joi.conversation.v1` storage key. The browser can retain 200 messages; requests
send only the latest 38 completed messages plus the current user message, within
the API limit. Actual model context is smaller and is selected by the model engine.
Assistant messages still come only from our own locally trained models.
