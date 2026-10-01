# Changelog

## [0.3.1] - 2026-10-01

### Added

- Deterministic English and Spanish greeting, identity, and capabilities intents in the assistant.
- Basic punctuation and surrounding whitespace normalization for conversational matches.
- Unit coverage for the requested conversation examples and matching behavior.

### Changed

- Bumped package version to 0.3.1.

## [0.3.0] - 2026-09-30

### Added

- Location-and-time scenario builder with deterministic synthetic historical analogs, replay safeguards, and nearest-AP geospatial resolution.
- Separate next-hour connection regressor alongside the backward-compatible LOW/HIGH classifier.
- Validation-calibrated split-conformal prediction intervals and capacity-use estimates.
- Deterministic tool-grounded assistant service on port 8001 and allowlisted aggregate dataset queries.
- Nine-section Streamlit navigation, scenario prediction, interactive synthetic AP map, and advanced feature form.
- Regression and scenario service tests, model metadata, and V3 documentation.

### Changed

- Model training/evaluation now reports classifier and regressor metrics on the existing temporal split.
- Removed generated HTML evaluation charts; evaluation writes compact PNG figures and JSON/CSV metrics.
- Set package version to 0.3.0.

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
