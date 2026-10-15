import unittest

import pandas as pd

from docmeta.evaluation import evaluate

TRUTH = pd.DataFrame({
    "File Name": ["a", "b", "c"],
    "Aggrement Value": ["6500", "4000", "9000"],
    "Aggrement Start Date": ["01.04.2010 ", "05.12.2008", "20.05.2007"],
    "Aggrement End Date": ["31.03.2011", "31.11.2009", "20.05.2008"],
    "Renewal Notice (Days)": ["15", None, "60"],
    "Party One": ["P C MATHEW ", "Balaji.R", "X"],
    "Party Two": ["L GOPINATH", "Kartheek R", "Y"],
})


class EvaluationTests(unittest.TestCase):
    def pred(self, **overrides):
        df = pd.DataFrame({
            "File Name": ["a", "b", "c"],
            "Aggrement Value": ["6500", "4000", "9000"],
            "Aggrement Start Date": ["01.04.2010", "05.12.2008", "20.05.2007"],
            "Aggrement End Date": ["31.03.2011", "31.11.2009", "20.05.2008"],
            "Renewal Notice (Days)": ["15", "", "60"],
            "Party One": ["P C MATHEW", "Balaji.R", "X"],
            "Party Two": ["L GOPINATH", "Kartheek R", "Y"],
        })
        for col, vals in overrides.items():
            df[col] = vals
        return df

    def scores(self, pred, **kw):
        s, errs = evaluate(pred, TRUTH, **kw)
        return {x.field: x for x in s}, errs

    def test_perfect_after_whitespace_cleanup(self):
        s, errs = self.scores(self.pred())
        self.assertTrue(all(x.recall == 1.0 for x in s.values()))
        self.assertEqual(len(errs), 0)

    def test_empty_truth_not_scored(self):
        s, _ = self.scores(self.pred())
        self.assertEqual(s["Renewal Notice (Days)"].total, 2)

    def test_miss_and_blank_prediction_count_as_false(self):
        pred = self.pred(**{"Party One": ["WRONG", "", "X"]})
        s, errs = self.scores(pred)
        self.assertEqual(s["Party One"].correct, 1)
        self.assertEqual(s["Party One"].total, 3)
        self.assertEqual(len(errs), 2)

    def test_strict_mode_sees_trailing_space(self):
        s, _ = self.scores(self.pred(), strict=True)
        self.assertLess(s["Aggrement Start Date"].recall, 1.0)

    def test_missing_prediction_row(self):
        pred = self.pred().iloc[:2]
        s, _ = self.scores(pred)
        self.assertEqual(s["Aggrement Value"].correct, 2)
        self.assertEqual(s["Aggrement Value"].total, 3)

    def test_accepts_corrected_column_spelling(self):
        pred = self.pred().rename(columns={"Aggrement Value": "Agreement Value"})
        s, _ = self.scores(pred)
        self.assertEqual(s["Aggrement Value"].recall, 1.0)


if __name__ == "__main__":
    unittest.main()
