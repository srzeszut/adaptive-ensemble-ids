class DummyTracker:
    emissions_kwh = 0.0

    def __init__(self, task_name: str = "task"):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class EnergyTracker:
    def __init__(self, task_name: str = "task"):
        from codecarbon import EmissionsTracker

        self.tracker = EmissionsTracker(
            project_name=task_name, log_level="error", save_to_file=False
        )
        self.emissions_kwh = 0.0

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, *_):
        self.tracker.stop()
        self.emissions_kwh = self.tracker.final_emissions_data.energy_consumed
