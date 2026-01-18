def load_system_identity(file_path="dawid_identity.txt") -> str:
    """Load the system identity (basic description of DAWID)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"System identity file '{file_path}' not found.")


def load_functions_description(file_path="dawid_functions.txt") -> str:
    """Load the available functions description."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""  # No functions available is OK for now


def load_full_system_prompt() -> str:
    """Load and combine system identity and function descriptions."""
    system_identity = load_system_identity()
    function_description = load_functions_description()

    if function_description:
        return f"{system_identity}\n\n{function_description}"
    else:
        return system_identity
