# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# class DataPreprocessor:
#     def __init__(self, split_ratio):
#         self.split_ratio = split_ratio
#         self.scaler = StandardScaler()

#     def preprocess(self, data):
#         X = data.iloc[:, :-1].values
#         y = data.iloc[:, -1].values
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.split_ratio)
#         X_train = self.scaler.fit_transform(X_train)
#         X_val = self.scaler.transform(X_val)
#         return X_train, X_val, y_train, y_val
