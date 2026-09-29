"""Project-specific exceptions with actionable messages."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected CSV cannot be located from the repository root."""


class DatasetValidationError(ValueError):
    """Raised when the dataset violates a critical data-contract rule."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when a persisted model artifact is missing."""


class PredictionError(ValueError):
    """Raised when inference inputs are incompatible with the trained pipeline."""
