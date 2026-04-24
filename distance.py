def get_direction(center_x, frame_width):
    left_limit = frame_width / 3
    right_limit = 2 * frame_width / 3

    if center_x < left_limit:
        return "Left"
    elif center_x < right_limit:
        return "Ahead"
    else:
        return "Right"


def estimate_distance(box_width):
    if box_width <= 0:
        return 0

    distance = round(500 / box_width, 1)
    return distance