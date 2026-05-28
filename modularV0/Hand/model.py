
"""
کلاس **GestureModel**
این کلاس مسئول طبقه‌بندی Gesture (نقش دست) بر اساس وضعیت
(باز/بسته) و درصد بسته‌شدن هر انگشت است.
به‌صورت ساده از `Finger_status`، `Finger_ang` و `Finger_Dist`
(فاصله‌های میان نقاط کلیدی) استفاده می‌کند و بر اساس قوانین
سفارشی، نام Gesture را برمی‌گرداند.

مهم‌ترین نکته:
- `Finger_status` یک دیکشنری است که کلیدهای آن نام انگشتان
  (thumb, index, middle, ring, pinky) و مقادیرشان
  «Extended» یا «Folded» می‌باشند.
- `Finger_Dist` یک دیکشنری است که فاصله‌ی بین دو انگشت
  را نگه می‌دارد؛ برای مثال `Finger_Dist[('Thumb','Index')]`
  مقدار فاصله‌ی انگشتان thumb و index است.
- `threshold` (در `__init__`) مقدار اولیه‌ای است که در زمان
  مقایسه‌ی درصد بسته‌شدن انگشتان (در متد `classify`) به‌کار می‌رود؛
  در این پروژه، مقدار 30 (درصد) به‌عنوان مرزی تعریف شده است.
"""

class GestureModel:
    def __init__(self, threshold=30):
        """
        پارامتر:
            threshold : درصد بسته‌ای که به‌صورت آستانه
            برای تشخیص Gesture استفاده می‌شود.
        """
        self.threshold = threshold
        # مرزی برای تشخیص «pinch» بر اساس فاصله‌ی Thumb‑Index
        self.Touch = 0.5 # اینو هم باید از config data بگیره

    def classify(self, Finger_status, Finger_ang, Finger_Dist):
        """
        ورودی‌ها:
            Finger_status : dict؛ کلید=انگشت، مقدار='Extended' یا 'Folded'
            Finger_ang    : dict؛ زاویه‌های انگشتان (برای آینده یا گسترش)
            Finger_Dist   : dict؛ فاصله‌ی میان انگشتان (برای pinch)
        خروجی:
            نام Gesture (مثلاً "pinch") یا "unknown" اگر مطابقتی پیدا نشد.
        """
        # ------------------------------------------------------------------
        # ۱. کشیدن انگشتان (pinch) – Thumb و Index بسته و فاصله‌ی کوتاه
        # ------------------------------------------------------------------
        if (Finger_status["Index"] == "Extended" and
            Finger_status["Thumb"] == "Extended" and
            Finger_Dist[('Thumb', 'Index')] < self.Touch):
            return "pinch"

        # ------------------------------------------------------------------
        # ۲. دست باز (open palm) – همه انگشتان باز
        # ------------------------------------------------------------------
        if all(v == "Extended" for v in Finger_status.values()):
            return "open_palm"

        # ------------------------------------------------------------------
        # ۳. مشت (fist) – همه انگشتان بسته
        # ------------------------------------------------------------------
        if all(v == "Folded" for v in Finger_status.values()):
            return "fist"

        # ------------------------------------------------------------------
        # ۴. thumbs_up – فقط thumb باز، دیگران بسته
        # ------------------------------------------------------------------
        if (Finger_status["Thumb"] == "Extended" and
            all(v == "Folded" for k, v in Finger_status.items() if k != "Thumb")):
            return "thumbs_up"

        # ------------------------------------------------------------------
        # ۵. thumb folded + دیگران بسته → (به‌نظر می‌رسد شرطی خالی است)
        # ------------------------------------------------------------------
        if (Finger_status["Thumb"] == "Folded" and
            all(v == "Folded" for k, v in Finger_status.items() if k != "Thumb")):
            return

        # ------------------------------------------------------------------
        # ۶. index_up – فقط index باز، دیگران بسته
        # ------------------------------------------------------------------
        if (Finger_status["Index"] == "Extended" and
            all(v == "Folded" for k, v in Finger_status.items() if k != "Index")):
            return "index_up"

        # ------------------------------------------------------------------
        # ۷. spiderman – thumb، index، pinky باز و ring، middle بسته
        # ------------------------------------------------------------------
        if (Finger_status["Middle"] == "Folded" and
            Finger_status["Ring"] == "Folded" and
            Finger_status["Pinky"] == "Extended" and
            Finger_status["Index"] == "Extended" and
            Finger_status["Thumb"] == "Extended"):
            return "spiderman"

        # ------------------------------------------------------------------
        # ۸. handgun – thumb، index، pinky باز، ring، pinky بسته
        # ------------------------------------------------------------------
        if (Finger_status["Ring"] == "Folded" and
            Finger_status["Pinky"] == "Folded" and
            all(v == "Extended" for k, v in Finger_status.items()
                if k not in ("Ring", "Pinky"))):
            return "handgun"

        # ------------------------------------------------------------------
        # ۹. در نهایت اگر هیچ‌یک از شرط‌های بالا برقرار نباشد
        # ------------------------------------------------------------------
        return "unknown"
