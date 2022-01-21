import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

import recourse as rs
    
class RecourseActions:
    def __init__(self):
        # import data
        df= pd.read_csv('./data/credit_processed.csv')
        self.y, self.X = df.iloc[:, 0], df.iloc[:, 1:]

        # train a classifier
        self.clf = load('./models/logistic_regression.joblib')

        yhat = self.clf.predict(self.X)

        # customize the set of actions
        A = rs.ActionSet(self.X)  ## matrix of features. ActionSet will set bounds and step sizes by default

        # specify immutable variables
        A['Married'].actionable = False

        # education level
        A['EducationLevel'].bounds = (0, 3)
        A['EducationLevel'].step_size = 1  ## set step-size to a custom value.
        A['EducationLevel'].step_direction = 1  ## force conditional immutability.
        A['EducationLevel'].step_type = "absolute"  ## force conditional immutability.

        # education level
        A['TotalMonthsOverdue'].step_size = 1  ## set step-size to a custom value.
        A['TotalMonthsOverdue'].step_type = "absolute"  ## discretize on absolute values of feature rather than percentile values
        A['TotalMonthsOverdue'].bounds = (0, 12)  ## set bounds to a custom value.


        A['MonthsWithLowSpendingOverLast6Months'].bounds = (0, 4)

        # can only specify properties for multiple variables using a list
        A[['Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60']].actionable = False
        A[['TotalMonthsOverdue', 'TotalOverdueCounts', 'HistoryOfOverduePayments']].actionable = False

        self.A = A

    def get_person(self, id=13):
        html_str = pd.DataFrame(self.X.loc[id,:]).to_html()
        return html_str
    
    def get_actions(self, id=13):
        # Person #13 is denied a loan (bad luck)
        x = self.X.values[[id]]
        yhat = self.clf.predict(x)[0]

        # Let's produce a list of actions that can change this person's predictions
        fs = rs.Flipset(x, action_set = self.A, clf = self.clf)
        fs.populate(enumeration_type = 'distinct_subsets', total_items = 10)
        html_str = fs.to_html()
        return html_str


def setup():
    recourse_actions = RecourseActions()
    return recourse_actions
