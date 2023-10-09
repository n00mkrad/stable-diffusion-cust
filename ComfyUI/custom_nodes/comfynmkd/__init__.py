from .nmkd_nodes import *
from .tiling import *

NODE_CLASS_MAPPINGS = {
    **nmkd_nodes.NODE_CLASS_MAPPINGS, 
    **tiling.NODE_CLASS_MAPPINGS, 
}
