import pandas as pd

from src import model_dispatcher, module, feature_selection, utils

# load df
x_train, y_train, x_test = module.load_dataset()
p_ids = x_test["PassengerId"].values

# preprocessing
x_train, x_test = module.data_preparation(x_train, x_test)
x_train = x_train.values.astype("float32")
x_test = x_test.values.astype("float32")
y_train = y_train.values.astype("float32").ravel()

# feature selection
mask = feature_selection.get_selected_features(x_train, y_train)
x_train = x_train[:, mask]
x_test = x_test[:, mask]

# model initialization
model = model_dispatcher.models["xgb_tuned"]

# predict
model.fit(x_train, y_train)
pred = model.predict(x_test)

# create submission
submission = pd.DataFrame(
    data=zip(p_ids, pred.astype(bool)), columns=["PassengerId", "Transported"]
)
submission.to_csv("output/submission.csv", index=False)
print("Done!")
