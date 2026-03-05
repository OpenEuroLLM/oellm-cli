from abc import ABC, abstractmethod

from oellm.task_groups import DatasetSpec


class BaseTask(ABC):
    """Abstract base class for evaluation task plugins.

    Subclasses represent a single logical evaluation task and provide
    the metadata needed for scheduling and dataset pre-download.

    This class is forward-looking: it is not yet consumed by the scheduling
    engine. Teams building T4.2–T4.5 integrations should subclass BaseTask
    to register new tasks without touching the YAML or core scheduling logic.

    Example (benchmark already in lmms-eval)::

        class VQAv2Task(BaseTask):
            @property
            def name(self) -> str:
                return "vqav2_val_all"

            @property
            def suite(self) -> str:
                return "lmms_eval"

            @property
            def n_shots(self) -> list[int]:
                return [0]

            @property
            def dataset_specs(self) -> list[DatasetSpec]:
                return [DatasetSpec(repo_id="HuggingFaceM4/VQAv2")]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical task name, used as the CSV task_path column value."""

    @property
    @abstractmethod
    def suite(self) -> str:
        """Evaluation suite identifier.

        Must be one of: ``lm_eval``, ``lighteval``, ``lmms_eval``.
        """

    @property
    @abstractmethod
    def n_shots(self) -> list[int]:
        """List of n-shot values to evaluate at."""

    @property
    def engine_task_name(self) -> str:
        """Task name as passed to the eval engine CLI.

        Defaults to ``name``. Override when the engine's registered task name
        differs from the canonical task name used in the CSV / YAML.
        """
        return self.name

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        """Dataset specifications for pre-download.

        Return an empty list if no pre-download is required (e.g. the dataset
        is already available on the compute nodes or pre-downloaded elsewhere).
        """
        return []
