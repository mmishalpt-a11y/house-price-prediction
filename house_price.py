import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
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




for column in df.select_dtypes(include=["object"]).columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )


print("\nData after encoding:")
print(df.head())




plt.figure(figsize=(8, 5))

sns.histplot(
    df["price"],
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()




plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="area",
    y="price"
)

plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()




X = df.drop(
    "price",
    axis=1
)

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

print("\nTraining Data:")
print(X_train.shape)

print("Testing Data:")
print(X_test.shape)




print("\n================================")
print("LINEAR REGRESSION")
print("================================")

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    linear_mse
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)

print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)



print("\n================================")
print("DECISION TREE")
print("================================")

decision_tree = DecisionTreeRegressor(
    random_state=42
)

print("Training Decision Tree...")

decision_tree.fit(
    X_train,
    y_train
)

print("Decision Tree trained successfully!")

decision_pred = decision_tree.predict(
    X_test
)

decision_mae = mean_absolute_error(
    y_test,
    decision_pred
)

decision_mse = mean_squared_error(
    y_test,
    decision_pred
)

decision_rmse = np.sqrt(
    decision_mse
)

decision_r2 = r2_score(
    y_test,
    decision_pred
)

print("\nDecision Tree Results:")

print("MAE:", decision_mae)
print("MSE:", decision_mse)
print("RMSE:", decision_rmse)
print("R2 Score:", decision_r2)


print("\n================================")
print("RANDOM FOREST")
print("================================")

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Training Random Forest...")

random_forest.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully!")

forest_pred = random_forest.predict(
    X_test
)

forest_mae = mean_absolute_error(
    y_test,
    forest_pred
)

forest_mse = mean_squared_error(
    y_test,
    forest_pred
)

forest_rmse = np.sqrt(
    forest_mse
)

forest_r2 = r2_score(
    y_test,
    forest_pred
)

print("\nRandom Forest Results:")

print("MAE:", forest_mae)
print("MSE:", forest_mse)
print("RMSE:", forest_rmse)
print("R2 Score:", forest_r2)


results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        decision_mae,
        forest_mae
    ],

    "MSE": [
        linear_mse,
        decision_mse,
        forest_mse
    ],

    "RMSE": [
        linear_rmse,
        decision_rmse,
        forest_rmse
    ],

    "R2 Score": [
        linear_r2,
        decision_r2,
        forest_r2
    ]
})


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    results.to_string(index=False)
)



plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x="Model",
    y="R2 Score"
)

plt.title("Model R2 Score Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("R2 Score")

plt.xticks(
    rotation=15
)

plt.tight_layout()

# Save comparison graph
plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




print("\n================================")
print("DECISION TREE VISUALIZATION")
print("================================")


tree_visual = DecisionTreeRegressor(
    random_state=42,
    max_depth=3
)


tree_visual.fit(
    X_train,
    y_train
)

print("Decision Tree Visualization Created!")



plt.figure(
    figsize=(25, 12)
)

plot_tree(
    tree_visual,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Decision Tree Regression",
    fontsize=18
)

plt.tight_layout()


plt.savefig(
    "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

print("Decision Tree saved as decision_tree.png")

plt.show()




print("\n================================")
print("DECISION TREE FEATURE IMPORTANCE")
print("================================")

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": decision_tree.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)


# Feature importance graph
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Decision Tree Feature Importance"
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "decision_tree_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()




best_model = results.loc[
    results["R2 Score"].idxmax()
]


print("\n================================")
print("BEST MODEL")
print("================================")

print(
    "Best Performing Algorithm:",
    best_model["Model"]
)

print(
    "Best R2 Score:",
    best_model["R2 Score"]
)\

print(
    "\nThe model with the highest R2 Score "
    "is the best-performing model."
)




print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")

print("\nGenerated files:")

print("1. decision_tree.png")
print("2. model_comparison.png")
print("3. decision_tree_feature_importance.png")
