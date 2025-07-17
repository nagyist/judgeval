from litellm import cost_per_token as _original_cost_per_token  # type: ignore

from judgeval.common.logger import judgeval_logger


def get_instance_prefixed_name(instance, class_name, class_identifiers):
    """
    Returns the agent name (prefix) if the class and attribute are found in class_identifiers.
    Otherwise, returns None.
    """
    if class_name in class_identifiers:
        class_config = class_identifiers[class_name]
        attr = class_config["identifier"]
        if hasattr(instance, attr):
            instance_name = getattr(instance, attr)
            return instance_name
        else:
            raise Exception(
                f"Attribute {attr} does not exist for {class_name}. Check your identify() decorator."
            )
    return None


def combine_args_kwargs(func, args, kwargs):
    """
    Combine positional arguments and keyword arguments into a single dictionary.

    Args:
        func: The function being called
        args: Tuple of positional arguments
        kwargs: Dictionary of keyword arguments

    Returns:
        A dictionary combining both args and kwargs
    """
    try:
        import inspect

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        args_dict = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                args_dict[param_names[i]] = arg
            else:
                args_dict[f"arg{i}"] = arg
        return {**args_dict, **kwargs}
    except Exception:
        return {**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs}


def cost_per_token(*args, **kwargs):
    try:
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = (
            _original_cost_per_token(*args, **kwargs)
        )
        if (
            prompt_tokens_cost_usd_dollar == 0
            and completion_tokens_cost_usd_dollar == 0
        ):
            judgeval_logger.warning("LiteLLM returned a total of 0 for cost per token")
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    except Exception as e:
        judgeval_logger.warning(f"Error calculating cost per token: {e}")
        return None, None
