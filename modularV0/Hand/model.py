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
        self.Touch=0.5

    def classify(self, Finger_status,Finger_ang,Finger_Dist):
        """
        ورودی: dict: 'thumb', 'index', ... => درصد بسته
        خروجی: نام گِیِست (یا 'unknown')
        """

        if Finger_status["Index"] == "Extended" and Finger_status["Thumb"] == "Extended" and Finger_Dist[('Thumb', 'Index')]<self.Touch :
            return "pinch"

        if all(v =="Extended" for v in Finger_status.values()):
            return "open_palm"


        if all(v =="Folded" for v in Finger_status.values()):
            return "fist"


        if Finger_status["Thumb"] =="Extended" and all(v =="Folded" for k, v in Finger_status.items() if k != "Thumb"):
            return "thumbs_up"


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



        if (Finger_status["Ring"] == "Folded" and
                Finger_status["Pinky"] == "Folded" and
                all(v == "Extended" for k, v in Finger_status.items()
                    if k != "Ring" and k != "Pinky")):
            return "handgun"

        return "unknown"
