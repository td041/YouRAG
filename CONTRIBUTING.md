# Contributing to YouRAG

## Setup

```bash
git clone https://github.com/td041/YouRAG.git
cd YouRAG
cp .env.example .env      # fill in GROQ_API_KEY
poetry install
docker compose up -d postgres qdrant redis
poetry run uvicorn src.api.main:app --reload --port 8000
```

## Code Rules

- **No new Python files** unless the task explicitly requires it
- **Type hints** on all new functions
- **No `print()`** in production code — use `logger` from `src.core.logger`
- **No bare `except:`** — always `except Exception:`
- **Graph store**: JSON only (`nx.node_link_data`) — never pickle (RCE risk)

## Singletons

```python
from src.core.database import db_instance   # VectorDatabase — do not re-instantiate
from src.core.config import settings        # Settings — do not re-instantiate
from src.core.redis_client import get_redis # Redis — lazy, returns None if unavailable
```

## Testing

```bash
# Run all unit tests (all mocked, no I/O)
poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Must pass before any PR
poetry run ruff check src/ tests/
poetry run bandit -r src/ -ll
```

### Mock pattern

```python
@pytest.fixture(autouse=True)
def mock_deps(mocker):
    mocker.patch("src.module.submodule.ClassName")
    mocker.patch("src.module.submodule.get_redis", return_value=None)

from src.module.submodule import MyClass  # import AFTER autouse fixture
```

## Commit style

```
feat: add X feature
fix: resolve Y bug
docs: update README
test: add coverage for Z
build: update Dockerfile
```

## Pull Request checklist

- [ ] `poetry run pytest tests/unit/` — all pass
- [ ] `poetry run ruff check src/ tests/` — no errors
- [ ] New features have unit tests
- [ ] No secrets or API keys in code
