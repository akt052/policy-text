OBJECT_TO_ID = {
    "ball": 0,
    "box": 1,
    "key": 2
}

COLOR_TO_ID = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "purple": 4,
    "grey": 5
}


def parse_mission(mission):

    words = mission.split()

    color = words[3]
    obj = words[4]

    return color, obj


def encode_object(obj, color):

    obj_id = OBJECT_TO_ID[obj]

    color_id = COLOR_TO_ID[color]

    return obj_id, color_id