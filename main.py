import pandas as pd
import seaborn as sns # this is a python graphing library
import matplotlib.pyplot as plt

teams = pd.read_csv('teams.csv')
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Show correlation with medals in numeric columns
print(teams.select_dtypes(include=['number']).corr()["medals"])

# # Create and display the scatter plot with regression line
# sns.set_style("whitegrid")
# plot = sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
# plot.set_axis_labels("Number of Athletes", "Medals Won")
# plt.title("Athletes vs Medals")
# plt.tight_layout()
# plt.show()

# #2
sns.set_style("whitegrid")
p = sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
p.set_axis_labels("Age", "Medals Won")
plt.title("Age vs Medals")
plt.tight_layout()
plt.show()


teams = teams[teams.isnull().any(axis=1)]
teams = teams.dropna()

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

predictors = ["athletes", "prev_medals"]
target = "medals"

reg.fit(train[predictors], train["medals"])

LinearRegression()

predictions = reg.predict(test[predictors])

test.loc[test["predicitions"] < 0, "predicitions"] = 0
test["predicitions"] = test["predicitions"].round()

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"], test["predicitions"])

print("Mean Absolute Error:", error)

errors = (test["medals"] - test["predicitions"]).abs() #finding each countries regression error

#in the end we see that those countries who send more  athletes to the olympics have lower error percentages
#in order for a more accurate prediction model...
# add in more predicitors and variables
# try different models like decision trees, random forests, or neural networks





