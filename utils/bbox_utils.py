def get_center_of_bbox(bbox):
    x1_new, y1_new, x2_new, y2_new = bbox
    return int((x1_new + x2_new) / 2), int((y1_new + y2_new) / 2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    x1_new, y1_new, x2_new, y2_new = bbox
    return int((x1_new + x2_new) / 2), int(y2_new)