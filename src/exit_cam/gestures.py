import math

TOUCH_THRESHOLD = 0.2
FINGERTIPS_TOUCHING = "FingertipsTouching"


def landmark_distance(point_a, point_b):
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def is_fingertips_touching(landmarks):
    """True when two hands have their thumb tips and their index tips close
    together. Close means within TOUCH_THRESHOLD times the size of the hand."""
    if len(landmarks) != 2:
        return False

    hand_one, hand_two = landmarks

    hand_size = landmark_distance(hand_one[0], hand_one[9])
    if hand_size == 0:
        return False

    thumb_distance = landmark_distance(hand_one[4], hand_two[4]) / hand_size
    index_distance = landmark_distance(hand_one[8], hand_two[8]) / hand_size

    return thumb_distance <= TOUCH_THRESHOLD and index_distance <= TOUCH_THRESHOLD


def hand_gestures(result):
    """The top gesture of every detected hand, as a list of name and score,
    plus an entry for each sign of our own that is being made."""
    gestures = [
        {"name": hand[0].category_name, "score": hand[0].score}
        for hand in result.gestures
        if hand
    ]

    if is_fingertips_touching(result.hand_landmarks):
        gestures.append({"name": FINGERTIPS_TOUCHING, "score": 1.0})

    return gestures


def is_trigger_sign(gestures, name, threshold):
    """True when either hand makes the sign called name with a score of at
    least threshold."""
    return any(
        gesture["name"] == name and gesture["score"] >= threshold
        for gesture in gestures
    )
