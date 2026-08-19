"""Exception hierarchy. Everything the agent raises on purpose inherits from AcaError."""


class AcaError(Exception):
    """Base class for all errors raised by the agent."""


class ConfigError(AcaError):
    """Missing or contradictory configuration."""


class BudgetExceeded(AcaError):
    """The run hit its iteration, time or dollar ceiling."""


class ToolError(AcaError):
    """A tool failed in a way the model is allowed to see and recover from."""


class SandboxViolation(ToolError):
    """A tool tried to touch something outside the workspace."""


class LLMError(AcaError):
    """The provider call failed after retries."""


class StoreError(AcaError):
    """The vector store is unreachable or returned something unusable."""
