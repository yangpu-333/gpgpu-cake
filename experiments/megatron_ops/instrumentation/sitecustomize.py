"""Opt-in Python startup hook for the Megatron RMSNorm capture wrapper."""

try:
    from rmsnorm_capture import install_from_environment

    install_from_environment()
except Exception as exc:  # Never obscure the user's original training error.
    import sys

    print(f"CAKE_RMSNORM_CAPTURE_INSTALL_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
