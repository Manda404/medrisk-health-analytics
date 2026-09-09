# Security policy

## Supported versions

Security fixes are applied to the latest released minor version.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or accidental exposure of patient data. Use GitHub's private vulnerability reporting for this repository. Include the affected version, reproduction steps, expected impact, and any suggested mitigation.

## Model artifact trust

The package loads Python pickle artifacts produced by its MLflow wrappers. Load artifacts only from a trusted and access-controlled MLflow registry. A malicious pickle file can execute code during deserialization.

## Patient data

Do not commit patient data, credentials, exported tables, or MLflow artifacts. Use Unity Catalog permissions, secret scopes, encrypted storage, and environment-specific service principals. Logs must not contain raw patient records or direct identifiers.
