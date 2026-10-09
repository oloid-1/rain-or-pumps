"""Entrypoint for Vercel and for `uvicorn main:app`: the simulator API and its UI."""

from simulator.api.app import app  # noqa: F401
