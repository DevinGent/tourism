import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# First we will create a list of dataframes.  We can start by getting a list of 
root_dir = "data"
frames = []
for dir_, _, files in os.walk(root_dir):
    for file in files:
        frames.append(pd.read_csv(root_dir+'\\'+file))

for frame in frames:
    frame.info()
    print()

# Let us check that the entity and code fields would match for New Zealand in each dataframe.
i=0
for frame in frames:
    print("Sorting by New Zealand vs NZL produces the same frame for index",i)
    print(frame[frame['Entity']=='New Zealand'].equals(frame[frame['Code']=='NZL']))
    print()
    i=i+1

# We will now restrict these dataframes to NZL and remove unnecessary columns.
i=0
first_year=3000
last_year=0
for frame in frames:
    frames[i]=frame[frame['Code']=='NZL'].drop(columns=['Entity','Code'])
    frame=frames[i]
    print("Index in list:",i)
    frame.info()
    min_year=min(frame['Year'].unique())
    max_year=max(frame['Year'].unique())
    if min_year<first_year:
        first_year=min_year
    if max_year>last_year:
        last_year=max_year
    print('First year listed:',min_year)
    print('Last year listed:',max_year)
    print()
    i=i+1
    
print("The years can range from {} to {}".format(first_year,last_year))

# We notice one of the dataframes (with index 4) has the unnecessary column 'Continent'.  We shall drop it.
frames[4]=frames[4].drop(columns=['Continent'])

# We are now ready to combine all of these dataframes together.
# We create a dataframe with one column, consisting of the years in question.
df=pd.DataFrame({'Year':[year for year in range(first_year,last_year+1)]})
print(df)

# We save the length of this dataframe.
df_length=df.shape[0]
print(df_length)

# It will help us to be able to fill in new cells of df by using the Year field to decide the row, and a column name to choose 
# the column.  To make this possible we will replace the original index with the year column (if we wanted to we could
# now drop the year column as it is redundent.)
df.set_index(df['Year'], inplace=True)
print(df)

# Testing how to strip a particular value from one of the smaller dataframes.
print(frames[4])
print(frames[4][frames[4]['Year']==2004]['Inbound tourism purpose (personal)'])
print(frames[4][frames[4]['Year']==2004]['Inbound tourism purpose (personal)'].values[0])


# For each of our smaller dataframes, we will try to copy over the non-year (the interesting) columns.
# First we begin iterating through the dataframes.
for frame in frames:
    # For each frame we only want to take the columns which are different from df: i.e. the non-year columns.
    for column in frame.drop(columns=['Year']).columns:
        # We create a new column of df, labeled the same as in frame, consisting of just nan values.
        df[column]=[np.nan]*df_length
        # We will now replace some of these nan values, where applicable, with the data from frame[column].
        for year in frame['Year'].unique():
            df.at[year,column]=frame[frame['Year']==year][column].values

# Let us see if this worked.
df.info()
print(df)
print(frames[-2])
print(frames[-1])
# Everything looks fine.

fig, axs= plt.subplots(2,4,sharex=True,figsize=(15,8))

for i in range(1,9):
    sns.lineplot(data=df,x='Year',y=df.columns[i],marker='o',ax=axs.flat[i-1])
    axs.flat[i-1].yaxis.label.set_size('small')
    axs.flat[i-1].vlines(x=[2001,2002,2003],label="The Lord of the Rings", ls='--',color='C1',
           ymin=0,
           ymax=[df[df['Year']==year][df.columns[i]].values for year in [2001,2002,2003]])
    axs.flat[i-1].vlines(x=[2012,2013,2014],label="The Hobbit", ls='--',color='C2',
           ymin=0,
           ymax=[df[df['Year']==year][df.columns[i]].values for year in [2012,2013,2014]])
plt.tight_layout()
plt.show()
# Here the orange lines represent released in the Lord of the Rings film series and the green releases in the Hobbit series.
