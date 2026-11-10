from auto import get_clean_dataset
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = get_clean_dataset()

print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

print(df[["stroke", "price"]].corr())

sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
plt.show()

df_body_style = df[["body-style", "price"]].groupby("body-style", as_index=False).mean()
print(df_body_style)

pair_wheel_base = df[["wheel-base", "price"]].dropna()
pearson_coef, p_value = stats.pearsonr(pair_wheel_base["wheel-base"], pair_wheel_base["price"])
print("Pearson correlation coefficient:", pearson_coef)
print("P-value:", p_value)

pair_horsepower = df[["horsepower", "price"]].dropna()
pearson_coef, p_value = stats.pearsonr(pair_horsepower["horsepower"], pair_horsepower["price"])
print("Pearson correlation coefficient:", pearson_coef)
print("P-value:", p_value)