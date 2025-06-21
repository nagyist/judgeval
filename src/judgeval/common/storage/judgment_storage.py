from judgeval.common.storage.storage import ABCStorage


class JudgmentStorage(ABCStorage):
    """
    Abstract base class for storage systems, responsible for storing judgment data.
    """

    def __init__(self):
        super().__init__()

    def save_trace(self, trace_data, trace_id, project_name):
        print(trace_data, trace_id, project_name)
        print("This is a placeholder for the save_trace method in JudgmentStorage.")
        return
