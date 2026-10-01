"""Domain exceptions exposed by the application layers."""


class DefectDetectorError(Exception):
    """Base exception for expected application errors."""


class DatasetError(DefectDetectorError):
    """Raised when a dataset violates the input contract."""


class InvalidImageError(DefectDetectorError):
    """Raised when an image cannot be decoded or is unsupported."""


class ModelArtifactError(DefectDetectorError):
    """Raised when a checkpoint is missing or inconsistent."""
