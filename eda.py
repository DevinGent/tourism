import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

# To make the code reproducible we set the seed.
random.seed(10)

# First we will look at the average length of stay.
df = pd.read_csv(r'data\10- average-length-of-stay.csv')
df.info()
print(df)

# What are the most frequent average stays?
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.histplot(df, x="Average length of stay")
plt.subplot(1,2,2)
sns.kdeplot(df, x="Average length of stay")
plt.suptitle("Across all years")
plt.show()

# We will do the same for the set of data before 2020 and after. First we split our dataframe into two parts.
post20=df[df['Year']>2019]
pre20=df[df['Year']<2020]
# We will create a figure with 4 plots, sharing x axes.
fig, axs=plt.subplots(nrows=2,ncols=2,figsize=(12,6), sharex=True)
sns.histplot(pre20, x="Average length of stay", ax=axs[0][0])
sns.histplot(post20, x="Average length of stay", ax=axs[1][0])
sns.kdeplot(pre20, x="Average length of stay",ax=axs[0][1])
sns.kdeplot(post20, x="Average length of stay",ax=axs[1][1])
plt.suptitle("Before and after 2019")
# We will add an annotation for each row to differentiate before and after 2020.
ax=axs[:,0][0]
ax.annotate("2019 and earlier", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
ax=axs[:,0][1]
ax.annotate("Post 2019", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
fig.tight_layout()
plt.show()

# We will also look at box plots for the entire dataset:
plt.figure(figsize=(6,7))
sns.boxplot([df["Average length of stay"],pre20["Average length of stay"],post20["Average length of stay"]])
plt.gca().set_xticklabels(['All years', '2019 and earlier', 'Post 2019'])
plt.gca().set_ylabel("Average length of stay")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(data=df,x='Year',y='Average length of stay')
plt.xlabel("Year")
plt.ylabel("Average length of stay")
plt.show()

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

sample_df=df[df['Entity'].isin(covid_nations)]
grouped=sample_df.groupby('Entity')['Average length of stay'].mean()
print(grouped)
print(grouped.rank(ascending=False))
nation_df=pd.DataFrame(grouped)
nation_df.reset_index(inplace=True)
nation_df['Rank']=nation_df['Average length of stay'].rank(method='min',ascending=False)
print(nation_df)
nation_df.info()
print(nation_df.sort_values('Rank'))


# Now let us examine the gdp dataset.
gdp_df=pd.read_csv(r'data\7- tourism-gdp-proportion-of-total-gdp.csv')
print(gdp_df)
gdp_df.info()

plt.figure(figsize=(12,6))
sns.lineplot(gdp_df[gdp_df['Entity']=='World'], x='Year',y="Tourism GDP as a proportion of Total")
plt.title("Worldwide Tourism")
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.histplot(gdp_df, x="Tourism GDP as a proportion of Total")
plt.subplot(1,2,2)
sns.kdeplot(gdp_df, x="Tourism GDP as a proportion of Total")
plt.suptitle("Across all years")
plt.show()

# We will do the same for the set of data before 2020 and after. First we split our dataframe into two parts.
post20=gdp_df[gdp_df['Year']>2019]
pre20=gdp_df[gdp_df['Year']<2020]
# We will create a figure with 4 plots, sharing x axes.
fig, axs=plt.subplots(nrows=2,ncols=2,figsize=(12,6), sharex=True)
sns.histplot(pre20, x="Tourism GDP as a proportion of Total", ax=axs[0][0])
sns.histplot(post20, x="Tourism GDP as a proportion of Total", ax=axs[1][0])
sns.kdeplot(pre20, x="Tourism GDP as a proportion of Total",ax=axs[0][1])
sns.kdeplot(post20, x="Tourism GDP as a proportion of Total",ax=axs[1][1])
plt.suptitle("Before and after 2019")
# We will add an annotation for each row to differentiate before and after 2020.
ax=axs[:,0][0]
ax.annotate("2019 and earlier", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
ax=axs[:,0][1]
ax.annotate("Post 2019", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
fig.tight_layout()
plt.show()

# We will also look at box plots for the entire dataset:
plt.figure(figsize=(6,7))
sns.boxplot([gdp_df["Tourism GDP as a proportion of Total"],
             pre20["Tourism GDP as a proportion of Total"],
             post20["Tourism GDP as a proportion of Total"]])
plt.gca().set_xticklabels(['All years', '2019 and earlier', 'Post 2019'])
plt.gca().set_ylabel("Tourism GDP as a proportion of Total")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(data=gdp_df,x='Year',y="Tourism GDP as a proportion of Total")
plt.xlabel("Year")
plt.ylabel("Tourism GDP as a proportion of Total")
plt.show()

# These graphs show that, after 2019, there are no longer countries with especially large proportions remaining.  
# Let us investigate this further by looking for countries which had over 15% of their GDP come from tourism at some point in time.
high_tourism=gdp_df[gdp_df['Tourism GDP as a proportion of Total']>15]
print(high_tourism)
high_tourism_nations=high_tourism['Entity'].unique()
print(high_tourism_nations)
print(gdp_df[gdp_df['Entity'].isin(high_tourism_nations)])
# The dataframe is missing enough information on these nations, with only Oceania, Guam, and Macao having data for years after 2019.
# Still, we can see a massive fall in all of these countries from 2019 to 2020:

plt.figure(figsize=(8,6))
sns.lineplot(data=gdp_df[gdp_df['Entity'].isin(['Oceania excluding Australia and New Zealand',
                                                'Macao',
                                                'Guam'])],x='Year',y="Tourism GDP as a proportion of Total", hue='Entity')
plt.show()
print("Between 2019 and 2020 we find that:")
for (place,shortplace) in zip(['Oceania excluding Australia and New Zealand','Macao','Guam'],
                              ['Oceania','Macao','Guam']):
    gdp2019=gdp_df[(gdp_df['Entity']==place)&(gdp_df['Year']==2019)]['Tourism GDP as a proportion of Total'].values[0]
    gdp2020=gdp_df[(gdp_df['Entity']==place)&(gdp_df['Year']==2020)]['Tourism GDP as a proportion of Total'].values[0]
    print(shortplace,"fell from {} to {}, a difference of {}.".format(gdp2019,gdp2020,gdp2019-gdp2020))
# In each case the drop is by more than half.

# More generally, let us look at what kind of drops happened post-covid.

# Let us calculate the change from the previous year (and the percentage change) in Tourism GDP.
def calculate_change(row):
    """This function takes a row from the dataframe gdp_df and calculates the 
    change in tourism from the previous year, as well as what percent of the previous year that change constitutes."""
    no_previous=False
    second_gdp=row['Tourism GDP as a proportion of Total']
    try:
        first_gdp=gdp_df[(gdp_df['Year']==row['Year']-1)&
                         (gdp_df['Entity']==row['Entity'])]['Tourism GDP as a proportion of Total'].values[0]
    except:
        no_previous=True
    if no_previous==True:
        change=np.nan
        change_pct=np.nan
    else:
        change=second_gdp-first_gdp
        change_pct=100*change/first_gdp
    return [change,change_pct]

gdp_df[['Change from Previous Year','Percent Change from Previous Year']]=gdp_df.apply(calculate_change,axis=1, result_type='expand')

print(gdp_df)
print(gdp_df.iloc[1])
print(calculate_change(gdp_df.iloc[1]))

# Let us calculate the averages in each factor by year across the countries.
year_averages=gdp_df.groupby('Year')[['Tourism GDP as a proportion of Total',
                                      'Change from Previous Year',
                                      'Percent Change from Previous Year']].mean()
print(year_averages)

# We will check if these categories are correlated.
print(gdp_df.corr(numeric_only=True))
# There does not seem to be significant correlation.


# Let us examine which countries had the largest tourism GDP (as a proportion of total) in a five year period before covid.
precovid_averages=gdp_df[gdp_df['Year'].isin([2015,2016,2017,2018,2019])].groupby('Entity')[['Tourism GDP as a proportion of Total',                                     
                                                                                            'Change from Previous Year',
                                                                                            'Percent Change from Previous Year']].mean()

print(precovid_averages.sort_values('Tourism GDP as a proportion of Total'))

# We will examine the set of countries with the highest (proportional) GDP and those with the lowest.
print(precovid_averages.sort_values('Tourism GDP as a proportion of Total').index)
top30_gdp=precovid_averages.sort_values('Tourism GDP as a proportion of Total').tail(30).index.tolist()
print("The 30 countries with the highest proportional GDPs shortly before Covid were:")
print(top30_gdp)
bottom_30gdp=precovid_averages.sort_values('Tourism GDP as a proportion of Total').head(30).index.tolist()
print("The 30 countries with the lowest proportional GDPs shortly before Covid were:")
print(bottom_30gdp)

# We will also look at which countries suffered the most significant (percentage) drop in 2020.
print(gdp_df[gdp_df['Year']==2020].dropna(subset=['Percent Change from Previous Year']).sort_values('Percent Change from Previous Year'))
biggest_drop2020=gdp_df[gdp_df['Year']==2020].dropna(subset=
                                                     ['Percent Change from Previous Year']).sort_values('Percent Change from Previous Year')['Entity'].head(30)

smallest_drop2020=gdp_df[gdp_df['Year']==2020].dropna(subset=
                                                     ['Percent Change from Previous Year']).sort_values('Percent Change from Previous Year')['Entity'].tail(30)
biggest_drop2020=biggest_drop2020.tolist()
smallest_drop2020=smallest_drop2020.tolist()
print("The countries with the largest percentage drop in proportional GDP were:")
print(biggest_drop2020)
print("The countries with the smallest percentage drop in proportional GDP were:")
print(smallest_drop2020)

vd=venn3([set(top30_gdp),set(bottom_30gdp),set(biggest_drop2020)],('Highest GDP (pre-Covid)','Lowest GDP (pre-Covid)','Largest Drop (post-Covid)'))
lbl = vd.get_label_by_id("C")
x, y = lbl.get_position()
lbl.set_position((x, y+.9))
plt.show()

# It seems that whole regions are listed as well. Let us see if this is true.
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(gdp_df.value_counts('Entity'))
# It is. Let us make a list of the regional entities.
print(gdp_df.loc[gdp_df['Code'].isnull()]['Entity'].unique())
regions=gdp_df.loc[gdp_df['Code'].isnull()]['Entity'].unique().tolist()
regions.append('World')
print(regions)
regional_gdp=gdp_df[gdp_df['Entity'].isin(regions)]
regional_gdp.drop(columns=['Code'], inplace=True)
print(regional_gdp)

plt.figure(figsize=(12,6))
sns.lineplot(data=regional_gdp,x='Year',y='Tourism GDP as a proportion of Total',hue='Entity')
plt.show()
print(regional_gdp[regional_gdp['Year'].isin([2019,2020])].drop(columns=['Change from Previous Year']))
# It seems that Latin America and the Caribbean sustained far less relative loss in GDP because of Covid than other regions.
# Central and Southern Asia sustained the most significant loss.