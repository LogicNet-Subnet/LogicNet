import random

def model_selector(model_pool, task_type="create_task"):
    """
    Select the model based on the task
    Input:
        model_pool: dict
            The pool of models available
        task_type: str
            The type of task to be performed. Can be either "create_task" or "score_task"
    """
    if task_type == "create_task" and "vllm" in model_pool:
        base_url = model_pool["vllm"][0]
        api_key = model_pool["vllm"][1]
        model = model_pool["vllm"][2]
    else:
        base_url = model_pool["openai"][0]
        api_key = model_pool["openai"][1]
        model = model_pool["openai"][2]
    return model, base_url, api_key