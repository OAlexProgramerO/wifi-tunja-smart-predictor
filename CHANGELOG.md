# Changelog

## [0.2.0] - 2026-09-29

### Changed

- Redesigned the Streamlit experience with explicit overview, exploration, analysis, prediction, performance, and about navigation.
- Removed runtime import-path mutations and rely on the editable `src/` package installation.
- Updated model metadata and generated evaluation reports from the current temporal holdout.
- Improved API, README, Spanish README, and technical documentation contracts and synthetic-data disclosures.

### Fixed

- Corrected the expected-missingness test to edit an existing datetime-indexed row.
- Made temporal split tests use consistent timestamps and added mixed-format parsing in split logic.
- Resolved Ruff import ordering and Black formatting findings.
- Bumped application/package version to 0.2.0.

## [0.1.0] - 2026-09-29

### Added

- Synthetic dataset generation
- Data validation
- Feature engineering
- Classification baseline models
- FastAPI prediction API
- Streamlit dashboard
- Automated tests
- Technical documentation
