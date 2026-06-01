VERSION := $(shell cat VERSION.txt)
TAG := v$(VERSION)

.PHONY: help sync lint typecheck test build check version tag release clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  sync          Install/update the uv environment' \
	  '  lint          Run ruff' \
	  '  typecheck     Run ty' \
	  '  test          Run pytest' \
	  '  build         Build sdist and wheel' \
	  '  check         Run lint, typecheck, tests, and build' \
	  '  version       Print VERSION.txt' \
	  '  tag           Create git tag v$$(cat VERSION.txt)' \
	  '  release       Run check, create tag, and push branch+tag' \
	  '  clean         Remove build/test caches'

sync:
	uv sync

lint:
	uv run -- ruff check

typecheck:
	uv run -- ty check

test:
	uv run -- pytest -q

build:
	uv build

check: lint typecheck test build

version:
	@cat VERSION.txt

tag:
	@uv run -- python -c 'from packaging.version import Version; import pathlib; v=pathlib.Path("VERSION.txt").read_text().strip(); assert str(Version(v)) == v, f"version must be normalized PEP 440: {v}"'
	@if ! git diff --quiet -- VERSION.txt pyproject.toml uv.lock; then \
	  echo 'version-related files have uncommitted changes; commit them before tagging' >&2; \
	  exit 2; \
	fi
	@if git rev-parse '$(TAG)' >/dev/null 2>&1; then \
	  echo 'tag $(TAG) already exists' >&2; \
	  exit 2; \
	fi
	git tag '$(TAG)'

release: check tag
	git push origin HEAD
	git push origin '$(TAG)'

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
