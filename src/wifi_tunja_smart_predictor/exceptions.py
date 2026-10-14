"""Project-specific exceptions with actionable messages."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected CSV cannot be located from the repository root."""


class DatasetValidationError(ValueError):
    """Raised when the dataset violates a critical data-contract rule."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when a persisted model artifact is missing."""


class ModelArtifactError(RuntimeError):
    """Raised when a persisted model artifact cannot be deserialized or used."""


class ModelMetadataError(ValueError):
    """Raised when model metadata is absent, malformed, or inconsistent."""


class PredictionError(ValueError):
    """Raised when inference inputs are incompatible with the trained pipeline."""


class LocationResolutionError(ValueError):
    """Raised when a synthetic location cannot be resolved safely."""


class ScenarioBuildError(ValueError):
    """Raised when historical context cannot construct a model-ready scenario."""


class AssistantQueryError(ValueError):
    """Raised when an assistant request or structured query is unsupported."""
