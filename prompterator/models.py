import importlib
from inspect import isclass
from pathlib import Path

from .constants import UNKNOWN_MODEL_NAME, ModelProperties, PrompteratorLLM

MODELS = {}
MODEL_INSTANCES = {}

current_dir = Path(__file__).parent.absolute()
MODEL_PATH = current_dir / "models"
for f in MODEL_PATH.glob("*.py"):
    module_name = f.stem
    module_spec = importlib.util.spec_from_file_location(module_name, f)

    if not module_name.startswith("_"):
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for cls in map(module.__dict__.get, module.__all__):
            if isclass(cls) and issubclass(cls, PrompteratorLLM):
                MODEL_INSTANCES[cls.name] = cls()
                MODELS[cls.name] = MODEL_INSTANCES[cls.name].properties
MODELS = {**MODELS, UNKNOWN_MODEL_NAME: ModelProperties(name=UNKNOWN_MODEL_NAME)}
