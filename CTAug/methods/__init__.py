from .GraphCL import GraphCL
from .GraphCL_AUG import GraphCL_AUG
from .InfoGraph import InfoGraph
from .JOAO import JOAO
from .JOAO_AUG import JOAO_AUG
from .JOAOv2 import JOAOv2
from .MVGRL import MVGRL
from .MVGRL_AUG import MVGRL_AUG
from .node_cls import CTAug_node


__all__ = ["GraphCL", "GraphCL_AUG",
           "JOAO", "JOAOv2", "JOAO_AUG",
           "MVGRL", "MVGRL_AUG",
           "InfoGraph", "CTAug_node"]