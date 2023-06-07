from xgboost import XGBClassifier


class XGBoostTM():
    def __init__(
        self,
        n_estimators,
        max_depth,
        learning_rate,
        objective
    ):
        self.model = XGBClassifier(
            n_estimators = n_estimators, 
            max_depth = max_depth, 
            learning_rate = learning_rate, 
            objective = objective
        )

    def fit(self, train_dataset, val_dataset):
        #TODO: unpack training dataset into two sets
        x, y = train_dataset
        self.model.fit(train_dataset, val_dataset)

        #TODO: do validation - or figure out where to put validation loop
