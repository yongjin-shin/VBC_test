REGISTRY = {}

from .basic_controller_corridor_vbc import BasicMAC_corridor
from .basic_controller_6h_vs_8z_vbc import BasicMAC_6h_vs_8z
from .basic_controller_8m_vbc import BasicMAC_8m

REGISTRY["basic_mac_corridor"] = BasicMAC_corridor
REGISTRY["basic_mac_6h_vs_8z"] = BasicMAC_6h_vs_8z
REGISTRY["basic_mac_8m"] = BasicMAC_8m