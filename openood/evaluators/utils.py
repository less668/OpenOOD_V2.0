from openood.evaluators.mos_evaluator import MOSEvaluator
from openood.utils import Config

from .ad_evaluator import ADEvaluator
from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .ece_evaluator import ECEEvaluator
from .fsood_evaluator import FSOODEvaluator
from .craft_ood_evaluator import CraftOODEvaluator
from .ood_evaluator import OODEvaluator
from .osr_evaluator import OSREvaluator
from .patchcore_evaluator import PatchCoreEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'craft_ood': CraftOODEvaluator,
        'fsood': FSOODEvaluator,
        'patch': PatchCoreEvaluator,
        'arpl': ARPLEvaluator,
        'ad': ADEvaluator,
        'mos': MOSEvaluator,
        'ece': ECEEvaluator,
        'osr': OSREvaluator
    }
    return evaluators[config.evaluator.name](config)
