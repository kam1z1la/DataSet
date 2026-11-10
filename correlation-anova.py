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