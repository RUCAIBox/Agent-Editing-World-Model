"""Merge generation settings without conflicting output-token aliases."""

from copy import deepcopy


def merge_generation_params(defaults, overrides):
    params = {**defaults, **overrides}
    for source in (overrides, defaults):
        selected = next(
            (key for key in ("max_completion_tokens", "max_tokens") if source.get(key) is not None),
            None,
        )
        if selected is not None:
            other = "max_tokens" if selected == "max_completion_tokens" else "max_completion_tokens"
            params.pop(other, None)
            params[selected] = source[selected]
            break
    if "extra_body" in defaults or "extra_body" in overrides:
        body = deepcopy(defaults.get("extra_body") or {})
        body.update(deepcopy(overrides.get("extra_body") or {}))
        params["extra_body"] = body
    return params
