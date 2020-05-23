import seaborn as sns

# source: https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/learn/lecture/5733426#overview
def plot_corr(df): 
    sns.heatmap(df.corr(), annot = True,fmt='.2f')