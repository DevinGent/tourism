import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To make the code reproducible we set the seed.
random.seed(10)

# First we will look at the average length of stay.
df = pd.read_csv(r'data\10- average-length-of-stay.csv')
df.info()
print(df)

# We will select 6 countries at random and graph the length of stay by year.
print(type(df['Entity'].unique()))
nation_list=random.sample(list(df['Entity'].unique()),6)
print(nation_list)

# It will be easier to compare the different nations if they share the same x axis (time range). Let us make a new dataframe
# which will have information for ALL years.
# Let's look at only those nations in nationlist.
print(df[df['Entity'].isin(nation_list)])
# Notice that Angola seems to have very few entries.  We will verify this.
print(df[df['Entity']=='Angola'])
# Let us rethink our plan.  We should only allow countries which will have information for pandemic years.
print(df[df['Year'].isin([2019,2020,2021,2022,2023])])

# We create an array of nations (as strings) which contain at least one year at 2019 or later.
covid_nations=df[df['Year'].isin([2019,2020,2021,2022,2023])]['Entity'].unique()
print(covid_nations)

# Now we can select a sample from this more limited list of countries.
nation_list=random.sample(list(covid_nations),6)
print(nation_list)
print(df[df['Entity'].isin(nation_list)])

sample_df=df[df['Entity'].isin(nation_list)]
start=sample_df['Year'].min()
stop=sample_df['Year'].max()
fig, axs= plt.subplots(2,3,figsize=(14,7))

for (nation,ax) in zip(nation_list,axs.flat):
    sns.lineplot(data=df[df['Entity']==nation],x='Year',y='Average length of stay',marker='o',ax=ax)
    #ax.set_title(nation)
    ax.set(title=nation,xlim=(start,stop))
plt.tight_layout()
plt.show()