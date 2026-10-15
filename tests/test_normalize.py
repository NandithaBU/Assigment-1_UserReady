import unittest

from docmeta.normalize import clean_text, normalize_date, normalize_int, normalize_result


class NormalizeTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  P C   MATHEW "), "P C MATHEW")
        self.assertIsNone(clean_text("   "))
        self.assertIsNone(clean_text(float("nan")))

    def test_int(self):
        self.assertEqual(normalize_int("6,500"), 6500)
        self.assertEqual(normalize_int(6500.0), 6500)
        self.assertEqual(normalize_int("15"), 15)
        self.assertIsNone(normalize_int("n/a"))
        self.assertIsNone(normalize_int(None))
        self.assertIsNone(normalize_int(True))

    def test_date_formats(self):
        self.assertEqual(normalize_date("1.4.2010"), "01.04.2010")
        self.assertEqual(normalize_date("20/05/2007"), "20.05.2007")
        self.assertEqual(normalize_date("01.04.2010 "), "01.04.2010")
        self.assertIsNone(normalize_date("May 2007"))
        self.assertIsNone(normalize_date("01.04.10"))

    def test_impossible_calendar_dates_are_kept(self):
        # Real labels contain these; they must not be rejected or "corrected".
        self.assertEqual(normalize_date("31.11.2009"), "31.11.2009")
        self.assertEqual(normalize_date("31.02.2011"), "31.02.2011")

    def test_result(self):
        out = normalize_result({
            "agreement_value": "6500", "agreement_start_date": "20-05-2007",
            "agreement_end_date": None, "renewal_notice_days": 15.0,
            "party_one": " A  B ", "party_two": "",
        })
        self.assertEqual(out["agreement_value"], 6500)
        self.assertEqual(out["agreement_start_date"], "20.05.2007")
        self.assertIsNone(out["agreement_end_date"])
        self.assertEqual(out["renewal_notice_days"], 15)
        self.assertEqual(out["party_one"], "A B")
        self.assertIsNone(out["party_two"])


if __name__ == "__main__":
    unittest.main()
