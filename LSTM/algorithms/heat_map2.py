import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# df = pd.read_excel(r"new.csv")   # pd.read_excel().set_index('R2')
df = np.loadtxt(r"new.csv",delimiter=',',unpack=False)
fig, ax = plt.subplots(figsize = (5,5))

sns.heatmap(df,annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap=None)

plt.show()  # 画出来的图太难看了
