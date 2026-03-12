"""Agent package exports with lazy loading."""

__all__ = ["construct_agent", "FunctionCallAgent"]


def construct_agent(*args, **kwargs):
    from .agent import construct_agent as _construct_agent

    return _construct_agent(*args, **kwargs)


def __getattr__(name):
    if name == "FunctionCallAgent":
        from .agent import FunctionCallAgent

        return FunctionCallAgent
    if name == "construct_agent":
        return construct_agent
    raise AttributeError(name)

