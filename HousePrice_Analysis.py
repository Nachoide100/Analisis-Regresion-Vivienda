
import pandas as pd
from scipy.stats import trim_mean, stats
from matplotlib import pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


filename = ("data/data_science_housing_1000.csv")
df = pd.read_csv(filename)
df.info()

#-- CHAPTER 1 --
#Stimates of location of SalePrice
mean_sale_price = df["SalePrice"].mean()
print(f"The sale price mean is {mean_sale_price}")

median_sale_price = df["SalePrice"].median()
print(f"The sale price median is {median_sale_price}")

trim_mean_sale_price = trim_mean(df["SalePrice"], 0.1)
print(f"The trimmed mean of sale price is {trim_mean_sale_price}")

#Median and mean are separated -> Outliers

#Plots
#Histogram
ax = df["SalePrice"].plot.hist(figsize=(4, 4), bins=30)
ax.set_xlabel("SalePrice")
plt.savefig("visualization/saleprice_distribution.png")
plt.show()

#Density plot
ax = df["SalePrice"].plot.hist(density=True, figsize = (4, 4), bins=30)
df["SalePrice"].plot.density(ax=ax)
ax.set_xlabel("SalePrice")
plt.show()


#We conclude the same. Not normal distribution. Outliers

#Find key predictors
#Calculate the correlation matrix
# 1. Create a list of all your numeric predictor columns + the target
numeric_columns = [
    'OverallQual',
    'YearBuilt',
    'GrLivArea',
    'FullBath',
    'BedroomAbvGr',
    'GarageCars',
    'SalePrice'  # Include your target variable
]

# 2. Create a new DataFrame containing ONLY these numeric columns
numeric_df = df[numeric_columns]

corr_matrix = numeric_df.corr()
print(corr_matrix)
#Key Predictors:
#GrLivArea
#GarageCars
#OverallQual

#Plot
# 1. Set the figure size
plt.figure(figsize=(10, 8))

# 2. Create the heatmap
sns.heatmap(
    corr_matrix,     # The data to plot
    annot=True,      # Show the correlation numbers in each cell
    fmt='.2f',       # Format the numbers to 2 decimal places
    cmap='coolwarm', # Use a color palette (blue=negative, red=positive)
    linewidths=0.5   # Add lines between cells
)

# 3. Add a title
plt.title('Correlation Matrix of Housing Features')
plt.savefig("visualization/correlation_matrix.png")
plt.show()

#The Key predictors are GrLivArea and GarageCars
#Scatter plots for the key predictors
#GrLivArea
ax = df.plot.scatter(x="GrLivArea", y="SalePrice")
ax.set_xlabel("Garden Area")
ax.set_ylabel("SalePrice")
plt.show()
#YearBuilt
ax = df.plot.scatter(x="YearBuilt", y="SalePrice")
ax.set_xlabel("Year Built")
ax.set_ylabel("SalePrice")
plt.show()

#Boxplot for categorical variables
#OverallQUal
ax = df.boxplot(by="OverallQual", column="SalePrice")
ax.set_xlabel("Overall Qualification")
ax.set_ylabel("SalePrice")
plt.show()

#Central Air
ax = df.boxplot(by="CentralAir", column="SalePrice")
ax.set_xlabel("Central Air")
ax.set_ylabel("SalePrice")
plt.show()

#GarageCars
ax = df.boxplot(by="GarageCars", column="SalePrice")
ax.set_xlabel("Garage Cars")
ax.set_ylabel("SalePrice")
plt.show()

#Neighborhood
ax = df.boxplot(by="Neighborhood", column="SalePrice")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("SalePrice")
plt.show()

#-- PHASE 2 --
#QQ Plot
#QQPlot
stats.probplot(df["SalePrice"], dist="norm", plot=plt)
plt.show()

#Convert SalePrice to normal distribution
# --- Step 1: Apply the Log-Transform ---
# Create a new column 'SalePrice_Log' using the NumPy log function
df['SalePrice_Log'] = np.log(df['SalePrice'])


# 1. Plot the new Histogram and Density Plot
plt.figure(figsize=(7, 5))
ax1 = df['SalePrice_Log'].plot.hist(density=True, bins=30, alpha=0.7, label='Histogram')
df['SalePrice_Log'].plot.density(ax=ax1, label='Density Plot', linewidth=2)
plt.title('Distribution of Log-Transformed SalePrice')
plt.xlabel('Log(SalePrice)')
plt.legend()
plt.show()

# 2. Plot the new QQ-Plot
plt.figure(figsize=(6, 5))
stats.probplot(df['SalePrice_Log'], dist="norm", plot=plt)
plt.title('QQ-Plot of Log-Transformed SalePrice')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Log(SalePrice) Quantiles')
plt.grid(True)
plt.savefig("visualization/LogSalePrice_QQplot.png")
plt.show()

#We've create a new column SalePrice_log, which is the natural log of SalePrice.

#Find the neighborhood with the highest median
highest_median = df.groupby("Neighborhood")["SalePrice"].median()
print(highest_median)

#Bootstrap to obtain the confidence intervals
expensive_neighborhood = df[df["Neighborhood"] == "NridgHt"]["SalePrice"]
n_iterations = 1000
bootstrap_medians = []

for i in range(n_iterations):
    fake_sample = expensive_neighborhood.sample(frac=1, replace=True)
    fake_median = fake_sample.median()
    bootstrap_medians.append(fake_median)

lower_bound = np.percentile(bootstrap_medians, 2.5)
upper_bound = np.percentile(bootstrap_medians, 97.5)
print(f"--- Bootstrap 95% CI for Sale Price in NridgHt ---")
print(f"Original Median: {expensive_neighborhood.median():.3f}")
print(f"Original Mean: {expensive_neighborhood.mean():.3f}")
print(f"Lower Bound:   {lower_bound:.3f}")
print(f"Upper Bound:   {upper_bound:.3f}")
"""
"""
#-- PHASE 3 --
#Hypothesis
#Houses with central Air have a statistically different SalePrice than houses without it
#Null hypothesis = there is no difference between in the sale price mean having cental air or not
#We perform a t-test

t_statistic, p_value = stats.ttest_ind(df[df["CentralAir"] == "Y"]["SalePrice_Log"],
                       df[df["CentralAir"] == "N"]["SalePrice_Log"],
                       equal_var=False)
print("\n--- T-Test Results (Central Air vs. SalePrice) ---")
print(f"T-Statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.2g}")

#As p_value < 0.05, we can reject the null hypothesis

#TEST ANOVA
#Null hypothesis: there is no difference in the mean SalePrice between any of the neighborhoods.
#We perform the test ANOVA
#We obtain the different neighborhoods
neighborhood_names = df["Neighborhood"].unique()

#Create the list of samples
price_samples = [
    df[df["Neighborhood"] == name]["SalePrice_Log"]
    for name in neighborhood_names
]

#Perform the test
f_stat, p_value = stats.f_oneway(*price_samples)
# Step 4 (Improved): Interpret the result
print("\n--- ANOVA Test Results (Neighborhood vs. Log-SalePrice) ---")
print(f"F-Statistic (on log-data): {f_stat:.3f}")
print(f"P-Value on log data: {p_value:.3f}")

#As p-value <0.001, we can reject the null hypothesis

#CHi - square test between two categorical variables
#Null hypothesis: there is no relationship between CentralAir and OverallQUal
#Create a contingency table
contingency_table = pd.crosstab(
    df["CentralAir"],
    df["OverallQual"]
)

print("---Contingency Table---")
print(contingency_table)

#Perform the chi-square test
chi2_stat, p_value, dof, expected_freqs = stats.chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi-Square Statistic: {chi2_stat:.3f}")
print(f"P-value: {p_value:.3f}")
# -- PHASE 4 --
#Create the prediction model
#First, we must adjust the categorical columns
#1. Create the CentraAir_n column (It's binary)
df["CentralAir_n"] = np.where(df["CentralAir"] == "Y", 1, 0)
central_air_dummie = df["CentralAir_n"]
print(df[["CentralAir", "CentralAir_n"]].head())

#2. Get dummies for the neighborhood column
neighborhood_dummies = pd.get_dummies(df["Neighborhood"], prefix="Neigh")
#Avoid the trap
neighborhood_dummies_clean = pd.get_dummies(
    df["Neighborhood"],
    prefix="Neigh",
    drop_first=True
)
#Convert the columns to integers
neighborhood_dummies_clean = neighborhood_dummies_clean.astype(int)
#Join the new dummies back to our main DataFrame
df = pd.concat([df, neighborhood_dummies_clean], axis=1)
#Remove the original column
df = df.drop("Neighborhood", axis=1)
#Check the final work
print("\n--- Final DataFrame ready for modeling ---")
print(df.head())

#2. Get dummies for the GarageCars column
garage_cars_dummies = pd.get_dummies(df["GarageCars"], prefix="Garage")
#Avoid the trap
garage_cars_dummies_clean = pd.get_dummies(
    df["GarageCars"],
    prefix="Garage",
    drop_first=True
)
#Convert the columns to integers
garage_cars_dummies_clean = garage_cars_dummies_clean.astype(int)
#Join the new dummies back to our main DataFrame
df = pd.concat([df, garage_cars_dummies_clean], axis=1)
#Remove the original column
#df = df.drop("GarageCars", axis=1)
#Check the final work
print("\n--- Final DataFrame ready for modeling ---")
print(df.head())

#2. Get dummies for the FullBath column
full_bath_dummies = pd.get_dummies(df["FullBath"], prefix="Bath")
#Avoid the trap
full_bath_dummies_clean = pd.get_dummies(
    df["FullBath"],
    prefix="Bath",
    drop_first=True
)
#Convert the columns to integers
full_bath_dummies_clean = full_bath_dummies_clean.astype(int)
#Join the new dummies back to our main DataFrame
df = pd.concat([df, full_bath_dummies_clean], axis=1)
#Remove the original column
df = df.drop("FullBath", axis=1)
#Check the final work
print("\n--- Final DataFrame ready for modeling ---")
print(df.head())

#Create the dfs for the multiple linear regression
#Create the Y variable
Y = df["SalePrice_Log"]
#Predictors columns
numeric_predictor_columns = df[['OverallQual', 'GrLivArea', 'YearBuilt']]

X = pd.concat([
    numeric_predictor_columns,
    neighborhood_dummies_clean,
    central_air_dummie,
    garage_cars_dummies_clean,
    full_bath_dummies_clean], axis=1)

#Check the work
print("--- Y (Target) ---")
print(Y.head())
print("\n--- X (Predictors) ---")
print(X.head())
print("\n--- X dtypes ---")
X.info()

#Perform the test
# Split the data: 80% for training, 20% for testing
# random_state=42 ensures you get the same "random" split every time
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# 1. statsmodels requires you to add the intercept (const) term manually
X_train_const = sm.add_constant(X_train)

# 2. Create the OLS (Ordinary Least Squares) model
#    We are modeling y_train ~ X_train_const
model = sm.OLS(y_train, X_train_const)

# 3. Fit the model to the training data
mlr_model = model.fit()

# Print the full regression summary
print(mlr_model.summary())

#Test the model
# 1. Add the constant to the X_test data
X_test_const = sm.add_constant(X_test)

# 2. Make predictions on the X_test data
log_predictions = mlr_model.predict(X_test_const)

# 3. --- CRITICAL STEP: Invert the Transformation ---
# Predictions are in Log-Dollars. We must convert them back.
dollar_predictions = np.exp(log_predictions)

# We must also convert the *actual* test values back
y_test_dollars = np.exp(y_test)

# 4. Calculate the Root Mean Squared Error (RMSE) in dollars
mse = mean_squared_error(y_test_dollars, dollar_predictions)
rmse = np.sqrt(mse)

print(f"\n--- Model Performance on Test Set ---")
print(f"The standard deviation of the sale price is {df["SalePrice"].std():.3f}")
print(f"Average Prediction Error (RMSE): ${rmse:,.2f}")

# 1. Get residuals and fitted values from the *training* model
residuals = mlr_model.resid
fitted_values = mlr_model.fittedvalues

# --- Plot 1: Residuals vs. Fitted Plot ---
# (Checks for Homoscedasticity)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=fitted_values, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values Plot')
plt.xlabel('Fitted Values (Log-Price)')
plt.ylabel('Residuals')
plt.savefig("visualization/residuals_fitted_values")
plt.show()

# --- Plot 2: QQ-Plot of Residuals ---
# (Checks for Normality of Residuals)
plt.figure(figsize=(6, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ-Plot of Model Residuals')
plt.show()

#UPGRADE MODEL
#We suppose that there is not a linear relation between the variables, so we will need a spline or a GAM
# Create a figure object for the plots
fig = plt.figure(figsize=(12, 10))
# The function 'plot_regress_exog' creates a 2x2 grid of diagnostic plots
# for a single exogenous (predictor) variable.
# We pass it:
#   1. The fitted model ('mlr_model')
#   2. The name of the predictor we want to check ('GrLivArea')
#   3. The figure to draw on ('fig')
sm.graphics.plot_regress_exog(mlr_model, exog_idx="GrLivArea", fig=fig)

#Add an overall title (optional)
plt.suptitle("Regression Diagnostics for GrLivArea", y=1.02, fontsize=14)
plt.show()

#GarageCars
ax = df.boxplot(by="GarageCars", column="SalePrice")
ax.set_xlabel("Garage Cars")
ax.set_ylabel("SalePrice")
plt.show()

stats.poisson.rvs(2, size=100)