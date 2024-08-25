JOINT_NAMES = {
    "smplxjoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye",
        "right_eye",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3",
    ],
    "smpljoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ],
    "guoh3djoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
}

SMPLX_JOINT_PAIRS = [
    (0, 1),  # pelvis to left_hip
    (0, 2),  # pelvis to right_hip
    (0, 3),  # pelvis to spine1
    (3, 6),  # spine1 to spine2
    (6, 9),  # spine2 to spine3
    (9, 12),  # spine3 to neck
    (12, 15),  # neck to head
    (1, 4),  # left_hip to left_knee
    (2, 5),  # right_hip to right_knee
    (4, 7),  # left_knee to left_ankle
    (5, 8),  # right_knee to right_ankle
    (7, 10),  # left_ankle to left_foot
    (8, 11),  # right_ankle to right_foot
    (12, 13),  # neck to left_collar
    (12, 14),  # neck to right_collar
    (13, 16),  # left_collar to left_shoulder
    (14, 17),  # right_collar to right_shoulder
    (16, 18),  # left_shoulder to left_elbow
    (17, 19),  # right_shoulder to right_elbow
    (18, 20),  # left_elbow to left_wrist
    (19, 21),  # right_elbow to right_wrist
]

INFOS = {
    "smplxjoints": {
        "LM": JOINT_NAMES["smpljoints"].index("left_ankle"),
        "RM": JOINT_NAMES["smpljoints"].index("right_ankle"),
        "LF": JOINT_NAMES["smpljoints"].index("left_foot"),
        "RF": JOINT_NAMES["smpljoints"].index("right_foot"),
        "LS": JOINT_NAMES["smpljoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["smpljoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["smpljoints"].index("left_hip"),
        "RH": JOINT_NAMES["smpljoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["smplxjoints"]) - 1,
    },
    "smpljoints": {
        "LM": JOINT_NAMES["smpljoints"].index("left_ankle"),
        "RM": JOINT_NAMES["smpljoints"].index("right_ankle"),
        "LF": JOINT_NAMES["smpljoints"].index("left_foot"),
        "RF": JOINT_NAMES["smpljoints"].index("right_foot"),
        "LS": JOINT_NAMES["smpljoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["smpljoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["smpljoints"].index("left_hip"),
        "RH": JOINT_NAMES["smpljoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["smpljoints"]),
    },
    "guoh3djoints": {
        "LM": JOINT_NAMES["guoh3djoints"].index("left_ankle"),
        "RM": JOINT_NAMES["guoh3djoints"].index("right_ankle"),
        "LF": JOINT_NAMES["guoh3djoints"].index("left_foot"),
        "RF": JOINT_NAMES["guoh3djoints"].index("right_foot"),
        "LS": JOINT_NAMES["guoh3djoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["guoh3djoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["guoh3djoints"].index("left_hip"),
        "RH": JOINT_NAMES["guoh3djoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["guoh3djoints"]),
    },
}
