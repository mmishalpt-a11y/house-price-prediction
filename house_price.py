import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




df = pd.read_csv("Housing.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())




df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(df.shape)


label_encoder = LabelEncoder()

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = label_encoder.fit_transform(df[column])



plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.show()



plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()



plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()




X = df.drop("price", axis=1)
y = df["price"]

print("\nInput Features:")
print(X.columns)

print("\nTarget:")
print(y.name)




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)



models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


results = []




for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])



results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ]
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results)




plt.figure(figsize=(8, 5))

sns.barplot(
    data=results,
    x="Model",
    y="R2 Score"
)

plt.title("Model R2 Score Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)

plt.ylim(0, 1)

plt.show()




best_model = results.loc[
    results["R2 Score"].idxmax()
]

print("\n==============================")
print("BEST MODEL")
print("==============================")

print(best_model)

print(
    "\nBest Performing Algorithm:",
    best_model["Model"]
)

print(
    "\nThe model with the highest R2 Score "
    "is the best-performing model."
)