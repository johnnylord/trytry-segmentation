import sys

from .classification import ClassAgent
from .segmentation import SegmentAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
