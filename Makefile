.PHONY: install install-dev test test-cov lint format format-check typecheck verify smoke sync-skills sync-skills-check

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

test-cov:
	pytest --cov=adalift --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy src/adalift

sync-skills:
	python scripts/sync_agent_skills.py

sync-skills-check:
	python scripts/sync_agent_skills.py --check

verify: sync-skills-check lint format-check typecheck test-cov

smoke:
	python scripts/smoke_train.py --steps 3 --batch-size 2
