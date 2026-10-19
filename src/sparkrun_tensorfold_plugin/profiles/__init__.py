"""Engine profiles: the contract a profile is written against, and the registry they join.

This package ships no profile. A profile is registered by the package that defines it, which is a
small sparkrun plugin whose ``register`` hook calls :func:`register_profile` (see
``docs/PROFILES.md``). The model types a profile is built from are in :mod:`.model` and the build
script helpers in :mod:`.scripts`.
"""

from __future__ import annotations

from sparkrun_tensorfold_plugin.profiles.model import Profile, ProfileError

REGISTRY: dict[str, Profile] = {}


def register_profile(profile: Profile) -> None:
    """Add a validated profile to the registry.

    Registering the same profile again is allowed, because sparkrun may load a plugin module more
    than once in one process. A different profile under a name that is taken is an error.
    """
    profile.validate()
    existing = REGISTRY.get(profile.name)
    if existing is not None and existing != profile:
        raise ProfileError(f"profile {profile.name} is already registered with different contents")
    REGISTRY[profile.name] = profile


def get_profile(name: str) -> Profile:
    """Return the named profile, or raise saying which profiles exist (possibly none)."""
    try:
        return REGISTRY[name]
    except KeyError:
        if not REGISTRY:
            raise ProfileError(
                f"unknown profile {name!r}; no profile is registered. A profile comes from its own "
                "sparkrun plugin package, which must be enabled alongside this one"
            ) from None
        available = ", ".join(sorted(REGISTRY))
        raise ProfileError(f"unknown profile {name!r}; available profiles: {available}") from None


__all__ = ["REGISTRY", "Profile", "ProfileError", "get_profile", "register_profile"]
