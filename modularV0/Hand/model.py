import json

class GestureModel:

    def __init__(self, config_path="gestures.json"):
        with open(config_path, "r") as f:
            self.gestures = json.load(f)

    def classify(self, Finger_status, Finger_ang, Finger_Dist):

        for gesture in self.gestures:

            matched = True

            # بررسی وضعیت انگشت‌ها
            for finger, state in gesture.get("fingers", {}).items():

                if Finger_status[finger] != state:
                    matched = False
                    break

            if not matched:
                continue

            # بررسی فاصله‌ها
            distance_rule = gesture.get("distance")

            if distance_rule:

                finger1, finger2 = distance_rule["pair"]

                if Finger_Dist[(finger1, finger2)] > distance_rule["max"]:
                    continue

            return gesture["name"]

        return "unknown"