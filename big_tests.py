# TODO: Probar con 10000 filas de la base de datos

import detecting_duplicates
from detecting_duplicates import Algorithm

import datetime
from datetime import timezone

import pandas as pd
import numpy as np

import unittest
import pdb

def smaller_than(s, max_val):
    vals = s.split(';')
    ret = [v for v in vals if int(v) <= max_val]
    if ret:
        return ";".join(ret)
    else:
        return np.nan

class TestMediumSizeDataFrames(unittest.TestCase):
    def setUp(self):
        original = pd.read_csv("./data/Firefox/mozilla_firefox.csv")
        train = pd.read_csv("./data/Firefox/train.csv")

        selection = ["Issue_id",
                     "Priority",
                     # "Component",
                     # "Duplicated_issue",
                     # "Title",
                     # "Description",
                     "Status",
                     "Resolution",
                     "Version",
                     "Created_time",
                     "Resolved_time"]

        data = original.loc[:500, selection]
        data = data.set_index('Issue_id')

        data.loc[:,'Created_time'] = pd.to_datetime(data.loc[:,'Created_time'], infer_datetime_format=True)
        data.loc[:,'Resolved_time'] = pd.to_datetime(data.loc[:,'Resolved_time'], infer_datetime_format=True)

        to_secs = (lambda t:(t - datetime.datetime(1970,1,1, tzinfo=timezone.utc)).total_seconds())
    
        data.loc[:,'Created_time'] = data.loc[:,'Created_time'].apply(to_secs)
        data.loc[:,'Resolved_time'] = data.loc[:,'Resolved_time'].apply(to_secs)
        eps = {"Created_time": 43200, "Resolved_time": 43200}

        train = train.set_index('Issue_id').loc[data.index].reset_index()
        train = train.dropna()

        max_val = int(train.loc[train.index[-1], 'Issue_id'])
        for i in train.index:
            train.loc[i,'Duplicate'] = smaller_than(train.loc[i,'Duplicate'], max_val)
        self.hom = detecting_duplicates.hom_model(data, train, [Algorithm.HOM]*4 + [Algorithm.DEV]*2, eps)
        self.hom.calc_thres(train)

    def test_real_duplicates(self):
        print("Test")
        score1 = self.hom.wjk(161749, 167183)
        score2 = self.hom.wjk(14871, 167183)
        biz_score = self.hom.wjk(14871, 167183)

        pdb.set_trace()
        
        self.assertEqual(score1 > self.t, True,
                         'Not finding real_duplicates')

if __name__ == "__main__":
    unittest.main()
