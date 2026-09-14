from typing import Any

class OverrideResolveMixin:
    """Adds a helper for resolving optional method arguments against a data class's own attributes."""

    def _resolve_override(
        self,
        default_name: str,
        value: Any,
        mask=None,
        required: bool = True,
    ):
        """
        Resolve an override:

        - value is None  -> use self.<default_name>
        - value is str   -> use self.<value>
        - otherwise      -> return value as-is (e.g. Tensor)

        If required=False and the attribute doesn't exist, return None.

        If mask is provided, it is applied ONLY when the value was sourced from self
        (i.e., when value is None or str).
        """
        # User directly provided a tensor
        if value is not None and not isinstance(value, str):
            return value

        # Determine attribute name to pull from self
        name = default_name if value is None else value

        if not hasattr(self, name):
            if required:
                raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
            return None

        out = getattr(self, name)

        if mask is not None and out is not None:
            out = out[mask]

        return out