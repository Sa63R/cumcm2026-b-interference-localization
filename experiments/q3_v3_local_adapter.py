"""Existing V3 official-protocol bridge, driven by the common local client."""
from types import SimpleNamespace


class BudgetClient:
    def __init__(self, client, maximum):
        self.client, self.maximum, self.actions = client, maximum, 1

    @property
    def state(self):
        return self.client.state

    def request(self, method, position, channel):
        if self.actions >= self.maximum - 1:
            raise RuntimeError("Common action budget exhausted")
        result = method(position, channel)
        self.actions += 1
        return result

    def measure(self, position, channel):
        return self.request(self.client.measure, position, channel)

    def clear(self, position, channel):
        return self.request(self.client.clear, position, channel)


def warmup():
    from experiments.q3_selected.q3_runtime import v
    v.warmup()


def run_v3_origin20(client, *, problem=3, max_actions=10000):
    if problem != 3:
        raise ValueError("This adapter is for Q3")
    from experiments.q3_selected.q3_runtime import run_policy
    client.enter()
    data = run_policy(BudgetClient(client, max_actions), "v3_origin20")
    client.exit()
    fields = dict(error=None, exit_error=None,
                  cleared_count=client.state.cleared_count,
                  virtual_time_s=client.state.virtual_time_s,
                  completion_certified_under_model=data["complete_channel_certificate"],
                  policy_report=data)
    return SimpleNamespace(**fields, as_dict=lambda: fields)
