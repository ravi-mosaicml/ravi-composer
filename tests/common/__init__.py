import types
from typing import List, Type

from tests.common.compare import deep_compare
from tests.common.datasets import (RandomClassificationDataset, RandomClassificationDatasetHparams, RandomImageDataset,
                                   configure_dataset_hparams_for_synthetic)
from tests.common.events import EventCounterCallback, EventCounterCallbackHparams
from tests.common.hparams import assert_is_constructable_from_yaml, assert_registry_contains_entry
from tests.common.markers import device, world_size
from tests.common.metrics import MetricSetterCallback
from tests.common.models import (SimpleConvModel, SimpleConvModelHparams, SimpleModel, SimpleModelHparams,
                                 configure_model_hparams_for_synthetic)
from tests.common.state import assert_state_equivalent


def get_all_subclasses_in_module(module: types.ModuleType, cls: Type) -> List[Type]:
    """Get all implementations of a class in a __module__ by scanning the re-exports from __init__.py"""
    return [x for x in vars(module).values() if isinstance(x, type) and issubclass(x, cls) and x is not cls]


__all__ = [
    "assert_state_equivalent",
    "RandomClassificationDataset",
    "RandomClassificationDatasetHparams",
    "RandomImageDataset",
    "configure_dataset_hparams_for_synthetic",
    "SimpleConvModel",
    "SimpleModel",
    "SimpleModelHparams",
    "SimpleConvModelHparams",
    "EventCounterCallback",
    "EventCounterCallbackHparams",
    "deep_compare",
    "device",
    "world_size",
    "configure_model_hparams_for_synthetic",
    "get_all_subclasses_in_module",
    "assert_is_constructable_from_yaml",
    "assert_registry_contains_entry",
    "MetricSetterCallback",
]
