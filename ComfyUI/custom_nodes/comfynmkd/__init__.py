from .nmkd_nodes import *
from .tiling import *

# from .nmkd_nodes import NODE_CLASS_MAPPINGS
# from .tiling import NODE_CLASS_MAPPINGS

# __all__ = ['NODE_CLASS_MAPPINGS']

NODE_CLASS_MAPPINGS = {
    **nmkd_nodes.NODE_CLASS_MAPPINGS, 
    **tiling.NODE_CLASS_MAPPINGS, 
}
