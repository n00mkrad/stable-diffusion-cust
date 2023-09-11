from custom_nodes.comfy_controlnet_preprocessors.nodes import edge_line, normal_depth_map, pose #, semseg, others
NODE_CLASS_MAPPINGS = {
    **edge_line.NODE_CLASS_MAPPINGS, 
    **normal_depth_map.NODE_CLASS_MAPPINGS, 
    **pose.NODE_CLASS_MAPPINGS, 
    #**semseg.NODE_CLASS_MAPPINGS, 
    #**others.NODE_CLASS_MAPPINGS
}