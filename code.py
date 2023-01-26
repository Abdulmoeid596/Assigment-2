
#import important libraries
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 



def data_transpose(filename: str):
    
    # Read the file into a pandas dataframe
    dataframe = pd.read_csv(filename)
    
    # Transpose the dataframe
    df_transposed = dataframe.transpose()
    
    # Populate the header of the transposed dataframe with the header information 
   
    # silice the dataframe to get the year as columns
    df_transposed.columns = df_transposed.iloc[1]
    # As year is now columns so we don't need it as rows
    df_transposed_year = df_transposed[0:].drop('year')
    
    # silice the dataframe to get the country as columns
    df_transposed.columns = df_transposed.iloc[0]
    
    # As country is now columns so we don't need it as rows
    df_transposed_country = df_transposed[0:].drop('country')
    
    return dataframe, df_transposed_country, df_transposed_year




# Passing filename to worldbank_data_transpose function 
orig_df, country_as_col, year_as_col = data_transpose('wb_dataset.csv')



# countries fresh water data over specfic years
# we need to extract data from our original data frame
urban_pop = orig_df[['country','year','urban_population']]

# drop the null values present in the dataset
urban_pop = urban_pop.dropna()


# data related to 1990 
no_data_1990 = urban_pop[urban_pop['year'] == 1990] 

# data related to 1995
no_data_1995 = urban_pop[urban_pop['year'] == 1995] 

# data related to 2000
no_data_2000 = urban_pop[urban_pop['year'] == 2000] 

# data related to 2005 
no_data_2005 = urban_pop[urban_pop['year'] == 2005] 

# data related to 2010 
no_data_2010 = urban_pop[urban_pop['year'] == 2010]

# data related to 2015 
no_data_2015 = urban_pop[urban_pop['year'] == 2015]

# data related to 2020 
no_data_2020 = urban_pop[urban_pop['year'] == 2020] 


style.use('ggplot')

# set fig size
plt.figure(figsize=(15,10))

# set width of bars
barWidth = 0.1

# plot bar charts
plt.bar(np.arange(no_data_1990.shape[0]),
        no_data_1990['urban_population'],
        color='blue', width=barWidth, label='1990')

plt.bar(np.arange(no_data_1995.shape[0])+0.2,
        no_data_1995['urban_population'],
        color='skyblue',width=barWidth, label='1995')

plt.bar(np.arange(no_data_2000.shape[0])+0.3,
        no_data_2000['urban_population'],
        color='green',width=barWidth, label='2000')

plt.bar(np.arange(no_data_2005.shape[0])+0.4,
        no_data_2005['urban_population'],
        color='olive',width=barWidth, label='2005')

plt.bar(np.arange(no_data_2010.shape[0])+0.5,
        no_data_2010['urban_population'],
        color='dodgerblue',width=barWidth, label='2010')

plt.bar(np.arange(no_data_2015.shape[0])+0.6,
        no_data_2015['urban_population'],
        color='slategray',width=barWidth, label='2015')

plt.bar(np.arange(no_data_2020.shape[0])+0.6,
        no_data_2020['urban_population'],
        color='red',width=barWidth, label='2020')


# show the legends on the plot
plt.legend()

# set the x-axis label
plt.xlabel('Country',fontsize=15)

# add title to the plot 
plt.title("Urban Population",fontsize=15)

# add countries names to the 11 groups on the x-axis
plt.xticks(np.arange(no_data_1990.shape[0])+0.2,
           ('United Arab Emirates', 'Armenia', 'Belgium', 'France',
       'Hong Kong SAR, China', 'Indonesia', 'Kenya', 'Nepal',
       'Saudi Arabia', 'Sweden', 'Eswatini', 'Tajikistan'),
           fontsize=10,rotation = 45)

# show the plot
plt.show()


# we want to see countries urban_population over the years
co2 = orig_df[['country','year','co2_emissions']]

# drop the null values present in the dataset
co2  = co2.dropna()


# ### Filter from specific year from 1990 to 2015


# data related to 1990
data_1990 = co2[co2['year'] == 1990]

# data related to 1995
data_1995 = co2[co2['year'] == 1995]

# data related to 2000
data_2000 = co2[co2['year'] == 2000]

# data related to 2005
data_2005 = co2[co2['year'] == 2005] 

# data related to 2010
data_2010 = co2[co2['year'] == 2010]

# data related to 2015 
data_2015 = co2[co2['year'] == 2015] 

# data related to 2020
data_2020 = co2[co2['year'] == 2020]



co2.country.unique()


# ### PLOT barplot


style.use('ggplot')

# set fig size
plt.figure(figsize=(15,10))

# set width of bars
barWidth = 0.1 

# plot bar charts
plt.bar(np.arange(data_1990.shape[0]),
        data_1990['co2_emissions'],
        color='goldenrod', width=barWidth, label='1990')

plt.bar(np.arange(data_1995.shape[0])+0.2,
        data_1995['co2_emissions'],
        color='blue',width=barWidth, label='1995')

plt.bar(np.arange(data_2000.shape[0])+0.3,
        data_2000['co2_emissions'],
        color='greenyellow',width=barWidth, label='2000')

plt.bar(np.arange(data_2005.shape[0])+0.4,
        data_2005['co2_emissions'],
        color='olive',width=barWidth, label='2005')

plt.bar(np.arange(data_2010.shape[0])+0.5,
        data_2010['co2_emissions'],
        color='dodgerblue',width=barWidth, label='2010')

plt.bar(np.arange(data_2015.shape[0])+0.6,
        data_2015['co2_emissions'],
        color='slategray',width=barWidth, label='2015')



# show the legends on the plot
plt.legend()

# set the x-axis label
plt.xlabel('Country',fontsize=15)

# add title to the plot 
plt.title("Co2 Emission",fontsize=15)

# add countries names to the 11 groups on the x-axis
plt.xticks(np.arange(data_2005.shape[0])+0.2,
           ('United Arab Emirates', 'Armenia', 'Belgium', 'France',
       'Indonesia', 'Kenya', 'Nepal', 'Saudi Arabia', 'Sweden',
       'Eswatini', 'Tajikistan'),
             fontsize=10,rotation = 45)

# show the plot
plt.show()


# making dataframe of Belgium data from the original dataframe
bel = orig_df[orig_df['country'] == 'Belgium']


# ### Implement a Function which removes Null values and return clean data



def remove_null_values(feature):
    return np.array(feature.dropna())


# ### For the Features Present In UAE DataFrame remove the null values 
# ### Print Each Features Size 



# Making dataframe of all the feature in the avaiable in 
# UAE dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(bel[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(bel[['co2_emissions']])

argicultural_land = remove_null_values(bel[['agricultural_land']])

nitrous_oxide = remove_null_values(bel[['nitrous_oxide']])

fresh_water = remove_null_values(bel[['fresh_water']])

arable_land = remove_null_values(bel[['arable_land']])

population = remove_null_values(bel[['population_growth']])

urban_pop = remove_null_values(bel[['urban_population']])

gdp = remove_null_values(bel[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))




# after removing the null values we will create datafram forBelgium data
bel = pd.DataFrame({'GreenHouse Gases': [greenhouse[x][0] for x in range(30)],
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(30)],
                                 'co2_emissions': [co2_emissions[x][0] for x in range(30)],
                                 'fresh_water': [fresh_water[x][0] for x in range(30)],
                                 'Nitrous Oxide': [nitrous_oxide[x][0] for x in range(30)],
                                 'Population': [population[x][0] for x in range(30)],
                                 'Urban_pop': [urban_pop[x][0] for x in range(30)],
                                 'GDP': [gdp[x][0] for x in range(30)],
                                })


# ### Correlation Heatmap of Belgium

# create correlation matrix
corr_matrix = bel.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title('Correlation Matrix Of Belgium')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()


# ### Create a dataframe contain only France data


# making dataframe of France data
fa_df = orig_df[orig_df['country'] == 'France']


# ### Remove Null values from Features

# Making dataframe of all the feature in the avaiable in 
# SA dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(fa_df[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(fa_df[['co2_emissions']])

argicultural_land = remove_null_values(fa_df[['agricultural_land']])

nitrous_oxide = remove_null_values(fa_df[['nitrous_oxide']])

fresh_water = remove_null_values(fa_df[['fresh_water']])

cereal_yield = remove_null_values(fa_df[['cereal_yield']])

arable_land = remove_null_values(fa_df[['arable_land']])

population = remove_null_values(fa_df[['population_growth']])

urban_pop = remove_null_values(fa_df[['urban_population']])

gdp = remove_null_values(fa_df[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('cereal_yield Length = '+str(len(cereal_yield)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))


# ### Create a new DataFrame for France data contain no null values


# after removing the null values we will create datafram for France data
fr_data = pd.DataFrame({'GreenHouse Gases': [greenhouse[x][0] for x in range(30)],
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(30)],
                                 'co2_emissions': [co2_emissions[x][0] for x in range(30)],
                                 'fresh_water': [fresh_water[x][0] for x in range(30)],
                                 'Nitrous Oxide': [nitrous_oxide[x][0] for x in range(30)],
                                 'Population': [population[x][0] for x in range(30)],
                                 'cereal_yield': [cereal_yield[x][0] for x in range(30)],
                                 'Urban_pop': [urban_pop[x][0] for x in range(30)],
                                 'GDP': [gdp[x][0] for x in range(30)],
                                })


# ### Correlation Heatmap of France


# create correlation matrix
corr_matrix = fr_data.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='Greens', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title('Correlation Matrix Of France ')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()


# ### Make dataframe of Saudi Arabia data from the original dataframe


# making dataframe of Saudi Arabia data from the original dataframe
sa_df = orig_df[orig_df['country'] == 'Saudi Arabia']


# ### For the Features Present In Saudi Arabia DataFrame remove the null values 
# ### Print Each Features Size 


# Making dataframe of all the feature in the avaiable in 
# Hong Kong dataframe passing it to remove null values function 
# for dropping the null values 
greenhouse = remove_null_values(sa_df[['greenhouse_gas_emissions']])

co2_emissions = remove_null_values(sa_df[['co2_emissions']])

argicultural_land = remove_null_values(sa_df[['agricultural_land']])

nitrous_oxide = remove_null_values(sa_df[['nitrous_oxide']])

fresh_water = remove_null_values(sa_df[['fresh_water']])

cereal_yield = remove_null_values(sa_df[['cereal_yield']])

arable_land = remove_null_values(sa_df[['arable_land']])

population = remove_null_values(sa_df[['population_growth']])

urban_pop = remove_null_values(sa_df[['urban_population']])

gdp = remove_null_values(sa_df[['GDP']])

# find the lenght of each feature size
# this will help us in creating dataframe 
# to avoid axis bound error in data frame creation
print('greenhouse Length = '+str(len((greenhouse)))) 
print('argicultural_land Length = '+str(len(argicultural_land))) 
print('nitrous_oxide  Length = '+str(len(nitrous_oxide))) 
print('co2_emissions Length = '+str(len(co2_emissions)))
print('fresh_water Length = '+str(len(fresh_water)))
print('cereal_yield Length = '+str(len(cereal_yield)))
print('population Length = '+str(len(population)))
print('urban_pop Length = '+str(len(urban_pop)))
print('gdp Length = '+str(len(gdp)))




# after removing the null values we will create datafram for France data
sa_data = pd.DataFrame({'GreenHouse Gases': [greenhouse[x][0] for x in range(30)],
                                 'Argicultural_land': [argicultural_land[x][0] for x in range(30)],
                                 'co2_emissions': [co2_emissions[x][0] for x in range(30)],
                                 'Nitrous Oxide': [nitrous_oxide[x][0] for x in range(30)],
                                 'Population': [population[x][0] for x in range(30)],
                                 'cereal_yield': [cereal_yield[x][0] for x in range(30)],
                                 'Urban_pop': [urban_pop[x][0] for x in range(30)],
                                 'GDP': [gdp[x][0] for x in range(30)],
                                })



# create correlation matrix of Saudi Arabia
corr_matrix = sa_data.corr()
plt.figure(figsize=(10,10))

# Plot the correlation matrix using imshow
plt.imshow(corr_matrix, cmap='Blues', interpolation='none')

# Add labels and adjust the plot
plt.colorbar()
plt.title("Correlation Heatmap of Saudi Arabia")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)


# Show the plot
plt.show()




# we want to see countries greenhouse_gas_emissions over the years
# we need to filter our original data frame to get specific fields
gg_data = orig_df[['country','year','greenhouse_gas_emissions']]

# drop the null values present in the dataset
gg_data = gg_data.dropna()


# ### Filter the Data For All the Countries 



uae = gg_data[gg_data['country'] == 'United Arab Emirates']
arm = gg_data[gg_data['country']== 'Armenia']
bel =  gg_data[gg_data['country'] == 'Belgium'] 
fr = gg_data[gg_data['country'] == 'France'] 
hk = gg_data[gg_data['country'] == 'Hong Kong SAR, China'] 
ind = gg_data[gg_data['country'] == 'Indonesia'] 
ken = gg_data[gg_data['country'] ==  'Kenya'] 
nep = gg_data[gg_data['country'] == 'Nepal'] 
sa = gg_data[gg_data['country'] == 'Saudi Arabia'] 
swe = gg_data[gg_data['country'] ==  'Sweden'] 
esw = gg_data[gg_data['country'] ==  'Eswatini'] 
taj = gg_data[gg_data['country']== 'Tajikistan'] 


# In[130]:


# set fig size
plt.figure(figsize=(10,10))

# set the line plot value on x-axis and y-axis by year and nitrous_oxide respectively
plt.plot(uae.year, uae.greenhouse_gas_emissions, '--',label='United Arab Emirates')
plt.plot(arm.year, arm.greenhouse_gas_emissions,'--',label='Armenia')
plt.plot(bel.year, bel.greenhouse_gas_emissions,'-',label='Belgium')
plt.plot(fr.year, fr.greenhouse_gas_emissions,'-',label='France')
plt.plot(hk.year, hk.greenhouse_gas_emissions,'--',label='Hong Kong SAR, China')
plt.plot(ind.year, ind.greenhouse_gas_emissions,'-',label='Indonesia')
plt.plot(ken.year, ken.greenhouse_gas_emissions,'--',label='Kenya')
plt.plot(nep.year, nep.greenhouse_gas_emissions,'--',label='Nepal')
plt.plot(sa.year, sa.greenhouse_gas_emissions,'-',label='Saudi Arabia')
plt.plot(swe.year, swe.greenhouse_gas_emissions,'--',label='Sweden')
plt.plot(esw.year, esw.greenhouse_gas_emissions,'--',label='Eswatini')
plt.plot(taj.year, taj.greenhouse_gas_emissions,'--',label='Tajikistan')

#Set the X-axis label and make it bold
plt.xlabel('Year',fontweight='bold')

#Set the Y-axis labe
plt.ylabel('Emission rate',fontweight='bold')

# set the title
plt.title("Greenhouse Gases")

# show the legends on the plot and place it on suitable position
plt.legend(bbox_to_anchor=(0.99,0.6),shadow=True)

#show the line plot
plt.show()




# we want to see countries GDP over the years
# we need to filter our original data frame to get specific fields
GDP = orig_df[['country','year','GDP']]

GDP = GDP.dropna()


# ### Filter the Data For All the Countries 



uae = GDP[GDP['country'] == 'United Arab Emirates']
arm = GDP[GDP['country']== 'Armenia']
bel =  GDP[GDP['country'] == 'Belgium'] 
fr = GDP[GDP['country'] == 'France'] 
hk = GDP[GDP['country'] == 'Hong Kong SAR, China'] 
ind = GDP[GDP['country'] == 'Indonesia'] 
ken = GDP[GDP['country'] ==  'Kenya'] 
nep = GDP[GDP['country'] == 'Nepal'] 
sa = GDP[GDP['country'] == 'Saudi Arabia'] 
swe = GDP[GDP['country'] ==  'Sweden'] 
esw = GDP[GDP['country'] ==  'Eswatini'] 
taj = GDP[GDP['country']== 'Tajikistan'] 


# ### Line Plot of GDP 


# set fig size
plt.figure(figsize=(10,10))

# set the line plot value on x-axis and y-axis by year and nitrous_oxide respectively
plt.plot(uae.year, uae.GDP, '--',label='United Arab Emirates')
plt.plot(arm.year, arm.GDP,'--',label='Armenia')
plt.plot(bel.year, bel.GDP,'--',label='Belgium')
plt.plot(fr.year, fr.GDP,'-',label='France')
plt.plot(hk.year, hk.GDP,'--',label='Hong Kong SAR, China')
plt.plot(ind.year, ind.GDP,'--',label='Indonesia')
plt.plot(ken.year, ken.GDP,'--',label='Kenya')
plt.plot(nep.year, nep.GDP,'--',label='Nepal')
plt.plot(sa.year, sa.GDP,'--',label='Saudi Arabia')
plt.plot(swe.year, swe.GDP,'--',label='Sweden')
plt.plot(esw.year, esw.GDP,'--',label='Eswatini')
plt.plot(taj.year, taj.GDP,'-',label='Tajikistan')

#Set the X-axis label and make it bold
plt.xlabel('Year',fontweight='bold')

# set the title
plt.title("GDP")

# show the legends on the plot and place it on suitable position
plt.legend(bbox_to_anchor=(0.99,0.6),shadow=True)

#show the line plot
plt.show()



