#!/usr/bin/env bash
# ============================================================================
# web_entrypoint.sh — Flask vs FastAPI dispatcher
#
# The Isitec stack ships two interchangeable backends sharing the same
# inference core: Flask (webapp/isitec_app) and FastAPI (webapp/isitec_api).
# Same `/api/*` surface, same dashboard, same UDP triggers. The FastAPI
# variant adds `/ws/video` (binary JPEG stream) + `/ws/stats` (500ms JSON
# tick) on top.
#
# Pick at container start by setting WEB_BACKEND in the compose env block
# (or deploy/.deployment.env, picked up by up.sh):
#
#   WEB_BACKEND=flask    (default) — webapp/isitec_app/app.py
#   WEB_BACKEND=fastapi           — webapp/isitec_api/app.py
#
# Aliases: 'api', 'uvicorn' → fastapi.  Anything else falls back to flask.
# ============================================================================
set -e

case "${WEB_BACKEND:-flask}" in
    fastapi|api|uvicorn)
        echo "▶ web_entrypoint: starting FastAPI (webapp/isitec_api/app.py)"
        exec python webapp/isitec_api/app.py
        ;;
    flask|"")
        echo "▶ web_entrypoint: starting Flask (webapp/isitec_app/app.py)"
        exec python webapp/isitec_app/app.py
        ;;
    *)
        echo "⚠ web_entrypoint: unknown WEB_BACKEND='${WEB_BACKEND}' — falling back to Flask"
        exec python webapp/isitec_app/app.py
        ;;
esac
