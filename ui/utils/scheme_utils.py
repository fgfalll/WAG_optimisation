SCHEME_DISPLAY_NAMES = {
    "wag": "WAG",
    "huff_n_puff": "Huff-n-Puff",
    "swag": "SWAG",
    "tapered": "Tapered Injection",
    "pulsed": "Pulsed Injection",
    "continuous": "Continuous Injection",
}


def format_scheme_display_name(scheme: str) -> str:
    """Convert scheme name to display format.

    Args:
        scheme: The scheme name (e.g., 'wag', 'huff_n_puff', 'continuous')

    Returns:
        Human-readable display name (e.g., 'WAG', 'Huff-n-Puff')
    """
    if scheme in SCHEME_DISPLAY_NAMES:
        return SCHEME_DISPLAY_NAMES[scheme]
    return scheme.replace("_", " ").title()