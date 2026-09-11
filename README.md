🏠 House Price Prediction Using Machine Learning
📌 Project Overview
This project focuses on predicting house prices using machine learning regression algorithms.

The project uses the Housing.csv dataset and applies data preprocessing, exploratory data analysis (EDA), visualization, model training, evaluation, and comparison.

Three machine learning algorithms are implemented:

Linear Regression
Decision Tree Regression
Random Forest Regression
The models are evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
The model with the highest R² Score is selected as the best-performing model.

🎯 Objectives
The main objectives of this project are:

Load and understand the housing dataset.
Check the dataset structure and missing values.
Remove duplicate records.
Convert categorical variables into numerical values.
Perform exploratory data analysis.
Analyze relationships between housing features and price.
Train multiple regression models.
Evaluate model performance using standard regression metrics.
Compare the performance of different algorithms.
Visualize the Decision Tree.
Analyze feature importance.
Identify the best-performing machine learning model.
📂 Project Structure
House-Price-Prediction/
│
├── Housing.csv
├── house_price_prediction.py
│
├── decision_tree.png
├── model_comparison.png
├── decision_tree_feature_importance.png
│
└── README.md

Files Description
File	Description
Housing.csv	Housing dataset used for training and testing
house_price_prediction.py	Main Python machine learning program
decision_tree.png	Visualization of the Decision Tree
model_comparison.png	R² score comparison of the models
decision_tree_feature_importance.png	Feature importance visualization
README.md	Project documentation

📊 Dataset
The project uses a housing dataset stored in:

Housing.csv

The dataset contains information about houses such as:

Area
Number of bedrooms
Number of bathrooms
Number of stories
Parking
Furnishing status
Other housing-related features
The target variable is:

price

The goal is to predict the price of a house based on the available features.

🛠️ Technologies Used
The project is implemented using Python.

Libraries
pandas
numpy
matplotlib
seaborn
scikit-learn

Main Libraries and Their Uses
Library	Purpose
Pandas	Data loading and manipulation
NumPy	Numerical calculations
Matplotlib	Data visualization
Seaborn	Statistical visualization
Scikit-learn	Machine learning and evaluation

⚙️ Machine Learning Workflow
The project follows the following workflow:

Load Dataset
     ↓
Explore Dataset
     ↓
Check Missing Values
     ↓
Remove Duplicates
     ↓
Encode Categorical Variables
     ↓
Exploratory Data Analysis
     ↓
Select Features and Target
     ↓
Train-Test Split
     ↓
Train Machine Learning Models
     ↓
Evaluate Models
     ↓
Compare Models
     ↓
Visualize Results
     ↓
Select Best Model

🔍 1. Loading the Dataset
The dataset is loaded using Pandas:

df = pd.read_csv("Housing.csv")

The first five rows, dataset shape, column names, and missing values are displayed.

🧹 2. Data Preprocessing
Removing Duplicate Records
Duplicate rows are removed using:

df.drop_duplicates(inplace=True)

This helps prevent duplicate observations from influencing the machine learning models.

Encoding Categorical Variables
Categorical columns are converted into numerical values using LabelEncoder:

for column in df.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )

This allows machine learning algorithms to process categorical information numerically.

📈 3. Exploratory Data Analysis
Several visualizations are created to understand the dataset.

House Price Distribution
A histogram with KDE is used to visualize the distribution of house prices.

sns.histplot(
    df["price"],
    kde=True
)

This helps understand how house prices are distributed.

Area vs House Price
A scatter plot is used to analyze the relationship between house area and price.

sns.scatterplot(
    data=df,
    x="area",
    y="price"
)

This can help identify whether larger houses generally have higher prices.

Correlation Heatmap
A correlation heatmap is generated using:

sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f"
)

The heatmap shows the correlation between numerical features and helps identify features that have stronger relationships with house prices.

🎯 4. Feature and Target Selection
The target variable is:

y = df["price"]

All remaining columns are used as input features:

X = df.drop(
    "price",
    axis=1
)

Therefore:

X → Input Features
y → House Price

✂️ 5. Train-Test Split
The dataset is divided into training and testing data using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

The split is:

80% Training Data
20% Testing Data
The random_state=42 ensures that the same split can be reproduced.

🤖 Machine Learning Models
1. Linear Regression
Linear Regression is used as a baseline regression algorithm.

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

The trained model predicts house prices on the testing dataset.

Linear Regression attempts to model the relationship between the input features and house price using a linear relationship.

2. Decision Tree Regression
A Decision Tree Regressor is trained using:

decision_tree = DecisionTreeRegressor(
    random_state=42
)

The Decision Tree learns a series of decision rules from the training data.

It can capture non-linear relationships between housing features and prices.

3. Random Forest Regression
A Random Forest Regressor is also trained:

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

The Random Forest combines multiple decision trees to produce a stronger prediction.

Using multiple trees can improve prediction performance and reduce the risk of relying on a single decision tree.

📏 Model Evaluation
Each model is evaluated using four metrics.

Mean Absolute Error (MAE)
mean_absolute_error(y_test, predictions)

MAE measures the average absolute difference between actual and predicted prices.

Lower MAE is better.

Mean Squared Error (MSE)
mean_squared_error(y_test, predictions)

MSE calculates the average squared difference between actual and predicted values.

Lower MSE is better.

Root Mean Squared Error (RMSE)
np.sqrt(mse)

RMSE is the square root of MSE.

It gives an estimate of the typical prediction error in the same units as the target variable.

Lower RMSE is better.

R² Score
r2_score(y_test, predictions)

R² measures how well the model explains the variation in house prices.

Generally:

Higher R² is better
1.0 indicates a perfect fit
Values closer to 0 indicate weaker explanatory performance
📊 Model Comparison
The results from all three models are stored in a Pandas DataFrame:

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    ...
})

The final comparison includes:

Model	MAE	MSE	RMSE	R² Score
Linear Regression	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution
Decision Tree	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution
Random Forest	Calculated during execution	Calculated during execution	Calculated during execution	Calculated during execution

The exact values depend on the Housing.csv dataset used when running the program.

📉 Model Comparison Visualization
The project creates an R² score comparison chart:

sns.barplot(
    data=results,
    x="Model",
    y="R2 Score"
)

The graph is saved as:

model_comparison.png

This provides a visual comparison of the predictive performance of the three algorithms.

🌳 Decision Tree Visualization
A separate Decision Tree with a maximum depth of 3 is created for visualization:

tree_visual = DecisionTreeRegressor(
    random_state=42,
    max_depth=3
)

The tree is visualized using:

plot_tree(
    tree_visual,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)

The resulting visualization is saved as:

decision_tree.png

The visualization makes the decision-making process of the regression tree easier to understand.

⭐ Feature Importance
The Decision Tree's feature importance values are extracted using:

decision_tree.feature_importances_

The features are sorted from highest to lowest importance.

The results are displayed in a bar chart and saved as:

decision_tree_feature_importance.png

Feature importance helps identify which housing attributes contributed most to the Decision Tree's predictions.

🏆 Best Model Selection
The best-performing model is selected based on the highest R² Score:

best_model = results.loc[
    results["R2 Score"].idxmax()
]

The program then displays:

Best Performing Algorithm
Best R2 Score

Therefore, the model with the highest R² Score is considered the best-performing model in this project.

📁 Generated Output Files
After successfully running the program, the following files are generated:

model_comparison.png
Contains the comparison of R² scores between:

Linear Regression
Decision Tree
Random Forest
decision_tree.png
Contains the visual representation of the Decision Tree.

decision_tree_feature_importance.png
Shows the relative importance of each feature according to the Decision Tree.

🚀 How to Run the Project
Step 1: Clone or Download the Project
Download the project files to your computer.

Make sure the following files are in the same directory:

Housing.csv
house_price_prediction.py

Step 2: Install Python
Make sure Python is installed on your system.

Check the installation with:

python --version

Step 3: Install Required Libraries
Run:

pip install pandas numpy matplotlib seaborn scikit-learn

Step 4: Run the Python Program
Run:

python house_price_prediction.py

The program will:

Load the dataset.
Preprocess the data.
Display exploratory visualizations.
Train three regression models.
Calculate evaluation metrics.
Compare model performance.
Generate visualization files.
Display the best-performing algorithm.
💻 Example Output
The program produces output similar to:

================================
MODEL COMPARISON
================================

             Model        MAE          MSE       RMSE   R2 Score
   Linear Regression       ...          ...        ...       ...
      Decision Tree        ...          ...        ...       ...
       Random Forest       ...          ...        ...       ...

The exact values depend on the dataset.

At the end, the program displays:

================================
BEST MODEL
================================

Best Performing Algorithm: Random Forest
Best R2 Score: ...

The actual best algorithm is determined automatically when the program runs.

📌 Key Findings
This project demonstrates that different regression algorithms can produce different prediction performances on the same housing dataset.

Linear Regression provides a simple baseline.
Decision Tree Regression can model non-linear relationships.
Random Forest Regression combines multiple decision trees to improve prediction robustness.
MAE, MSE, and RMSE measure prediction errors.
R² Score is used to compare how well each model explains the target variable.
Feature importance provides insight into which housing attributes are most influential in the Decision Tree model.
⚠️ Limitations
There are some limitations in the current implementation:

LabelEncoder is used for categorical features, which imposes numerical ordering on categories.
No feature scaling is applied.
Hyperparameter tuning is not performed.
The models are evaluated using a single train-test split.
Cross-validation is not used.
The project does not include a user interface for making individual house-price predictions.
The dataset may contain outliers that can affect model performance.
🔮 Future Improvements
The project can be improved by:

Using OneHotEncoder for categorical variables.
Applying feature scaling where appropriate.
Performing cross-validation.
Using GridSearchCV or RandomizedSearchCV for hyperparameter tuning.
Testing additional algorithms such as:
Gradient Boosting
XGBoost
Support Vector Regression
Extra Trees Regression
Performing outlier detection and treatment.
Adding feature engineering.
Creating a prediction interface using Streamlit or Flask.
Saving the best trained model using joblib or pickle.
Adding a function that allows users to enter house details and receive a predicted price.
📚 Conclusion
This project demonstrates a complete machine learning workflow for house price prediction.

The workflow starts with data loading and preprocessing, followed by exploratory data analysis, model training, evaluation, visualization, and model comparison.

By comparing Linear Regression, Decision Tree Regression, and Random Forest Regression using MAE, MSE, RMSE, and R² Score, the project identifies the algorithm that performs best on the given dataset.

The project provides a practical introduction to supervised machine learning, regression, data preprocessing, exploratory data analysis, and model evaluation.

