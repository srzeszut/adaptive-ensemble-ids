ABBREVIATIONS = {
    "STATIC_MAX": "MAX",
    "STATIC_MEAN": "MEAN",
    "STATIC_MEDIAN": "MED",
    "WEIGHTED_STATIC_MEAN": "WS",
    "WEIGHTED_STATIC_MAX": "WS-MAX",
    "DYNAMIC_HEUR_EQUAL_START": "H-EQ",
    "DYNAMIC_HEUR_CALIBRATED_START": "H-CAL",
    "DYNAMIC_META_EQUAL_START": "M-EQ",
    "DYNAMIC_META_CALIBRATED_START": "M-CAL",
    "DYNAMIC_HEUR_MAX_EQUAL": "H-MAX-EQ",
    "DYNAMIC_HEUR_MAX_CAL": "H-MAX-CAL",
    "DYNAMIC_META_MAX_EQUAL": "M-MAX-EQ",
    "DYNAMIC_META_MAX_CAL": "M-MAX-CAL",
    "RiverZScore": "RZS",
    "StreamCNN": "SCNN",
    "RiverKMeans": "RKM",
}


def abbreviate(name: str) -> str:
    name = name.removeprefix("BASELINE_")
    return ABBREVIATIONS.get(name, name)
