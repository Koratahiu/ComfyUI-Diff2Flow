"""
@author: Koratahiu
@title: ComfyUI-Diff2Flow
@nickname: ComfyUI-Diff2Flow
@description: Unofficial Implementation of Diff2Flow Method for ComfyUI.
"""

from .nodes import nodes_diff2flow

NODE_CLASS_MAPPINGS = {
    **nodes_diff2flow.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_diff2flow.NODE_DISPLAY_NAME_MAPPINGS,
}