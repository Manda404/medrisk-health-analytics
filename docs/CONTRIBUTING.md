# Contributing

## Development setup

```bash
poetry install
poetry run pre-commit install
make check
```

Use Python 3.12 through 3.14. Add focused tests for behavioral changes and keep optional ML dependencies behind lazy imports.

## Pull requests

1. Explain the problem and the resulting behavior.
2. Add or update tests for material logic changes.
3. Run `make check` before opening the pull request.
4. Keep patient data, credentials, local MLflow runs, and generated models out of commits.
5. Record user-visible changes in `CHANGELOG.md`.

Clinical thresholds and derived medical features require a source and review by a qualified clinical subject-matter expert before release.
