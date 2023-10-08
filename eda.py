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

# We return to making sure all the countries will share the same set of years.
# We will create a function that creates a more graph-friendly dataframe given a list of countries.

# First, we will see which countries have gaps (missing years) in their records.
gap_nations=[]
for nation in df['Entity'].unique():
    if len(df[df['Entity']==nation]['Year'])!=1+df[df['Entity']==nation]['Year'].max()-df[df['Entity']==nation]['Year'].min():
        gap_nations.append(nation)
print(gap_nations)
print(df[df['Entity']=='Algeria'])
# Quite a few have gaps.  We will take this into account when creating our function.

def make_graphable_df(dataframe,countries,factor):
    """Takes a dataframe, a list of countries, and the label of one column (factor) in the frame, then produces
    a new dataframe with columns ['Year']+countries, and each row represents the value of factor in each country for a 
    given year."""
    frame=dataframe[dataframe['Entity'].isin(countries)]
    start=frame['Year'].min()
    stop=frame['Year'].max()
    length=1+stop-start
    new_df=pd.DataFrame({'Year':[i for i in range(start,stop+1)]})
    new_df.set_index('Year',inplace=True)
    for country in countries:
        column=[]
        for i in range(start,stop+1):
            if i in frame[frame['Entity']==country]['Year'].values:
                column.append(frame[(frame['Entity']==country)&(frame['Year']==i)][factor].values[0])
            else:
                column.append(np.nan)   
        new_df[country]=column
    return new_df

# Let us test if this works:
test_df=make_graphable_df(df,['Algeria', 'Argentina'],'Average length of stay') 
print(test_df)
# And compare with just looking at those two countries in the original dataframe.
print(df[df['Entity'].isin(['Algeria', 'Argentina'])])
# It works well!

# Now we can use the random nations from nation_list.

sample_df=make_graphable_df(df,nation_list,'Average length of stay') 
print(sample_df)

fig, axs= plt.subplots(2,3,figsize=(14,7))

for (nation,ax) in zip(nation_list,axs.flat):
    sns.lineplot(data=sample_df,x=sample_df.index,y=nation,marker='o',ax=ax)
    #ax.set_title(nation)
    ax.set(title=nation,ylabel='Average length of stay')
plt.tight_layout()
plt.show()