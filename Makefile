# --- Variables ---
COMPOSE=docker compose
FRONTEND=frontend
BACKEND=backend

# --- Local (no Docker) ---
.PHONY: fe-install be-install dev-fe dev-be
fe-install:
	cd $(FRONTEND) && npm ci

be-install:
	python3 -m venv .venv && . .venv/bin/activate && \
	pip install --upgrade pip && pip install -r $(BACKEND)/requirements.txt

dev-fe:
	cd $(FRONTEND) && VITE_API_BASE=http://localhost:8000 npm run dev

dev-be:
	cd $(BACKEND) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# --- Docker ---
.PHONY: build up down logs sh-fe sh-be
build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

sh-fe:
	$(COMPOSE) exec frontend sh

sh-be:
	$(COMPOSE) exec backend sh
