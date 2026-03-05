from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    An adapter's sole responsibility is translating a model path/config into
    the engine-specific argument strings passed on the command line.  The
    scheduling engine reads these strings and injects them into the sbatch
    template — adapters never call the engines directly.

    This is the single integration point for adding new or proprietary models:
    implement the two abstract methods and the adapter works with any engine
    the platform supports.

    Example::

        class HFModelAdapter(BaseModelAdapter):
            def __init__(self, path: str, trust_remote_code: bool = True):
                self._path = path
                self._trust = trust_remote_code

            @property
            def model_path(self) -> str:
                return self._path

            def to_lm_eval_args(self) -> str:
                return f"pretrained={self._path},trust_remote_code={self._trust}"

            def to_lmms_eval_args(self) -> str:
                return f"pretrained={self._path}"
    """

    @property
    @abstractmethod
    def model_path(self) -> str | Path:
        """Path to the model weights or HuggingFace repo ID."""

    @abstractmethod
    def to_lm_eval_args(self) -> str:
        """Return the ``--model_args`` string for lm-eval-harness.

        Example: ``"pretrained=/path/to/model,trust_remote_code=True"``
        """

    @abstractmethod
    def to_lmms_eval_args(self) -> str:
        """Return the ``--model_args`` string for lmms-eval.

        Example: ``"pretrained=/path/to/model"``
        """
