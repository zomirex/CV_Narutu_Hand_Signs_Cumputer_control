# hand/model.py
class GestureModel:
    """
    طبقه‌بندی گِیِست بر اساس درصد بسته‌شدن انگشتان
    """

    def __init__(self, threshold=30):
        """
        threshold: درصد بسته برای انگشتان
        """
        self.threshold = threshold

    def classify(self, Finger_status,Finger_ang):
        """
        ورودی: dict: 'thumb', 'index', ... => درصد بسته
        خروجی: نام گِیِست (یا 'unknown')
        """
        # گِیِست “باز” (تمام انگشتان باز)
        if all(v =="Extended" for v in Finger_status.values()):
            return "open_palm"

        # گِیِست “فیس” (تمام انگشتان بسته)
        if all(v =="Folded" for v in Finger_status.values()):
            return "fist"

        # گِیِست “Thumbs Up”
        if Finger_status["Thumb"] =="Extended" and all(v =="Folded" for k, v in Finger_status.items() if k != "Thumb"):
            return "thumbs_up"

        # گِیِست “Thumbs Down” (thumb بسته و انگشتان باز)
        if Finger_status["Thumb"] =="Folded" and all(v =="Folded" for k, v in Finger_status.items() if k != "Thumb"):
            return


        if Finger_status["Index"] =="Extended" and all(v =="Folded" for k, v in Finger_status.items() if k != "Index"):
            return "index_up"

        if Finger_status["Middle"] =="Folded" and\
                Finger_status["Ring"] =="Folded" and\
                Finger_status["Pinky"] =="Extended" and\
                Finger_status["Index"] =="Extended" and\
                Finger_status["Thumb"] =="Extended" :
            return "spiderman"

        if (Finger_status["Index"] == "Extended" and
                Finger_status["Thumb"] == "Extended" and
                all(v == "Folded" for k, v in Finger_status.items()
                    if k != "Index" and k != "Thumb")):
            return "pinch"

        if (Finger_status["Ring"] == "Folded" and
                Finger_status["Pinky"] == "Folded" and
                all(v == "Extended" for k, v in Finger_status.items()
                    if k != "Ring" and k != "Pinky")):
            return "handgun"

        return "unknown"
