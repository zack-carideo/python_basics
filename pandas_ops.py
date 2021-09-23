import pandas as pd
import os
import numpy as np
from pydataset import data
from pandas_profiling import ProfileReport

# load a sample df to show pandas operations
df = data('movies')

'''describe data'''
df.describe()
df.info()
prof = ProfileReport(df, minimal=True)
prof.to_file(output_file='output.html')
os.remove('output.html')  # comment this out if you want to review the file

'''rename columns'''
df = df.rename(columns={'mpaa': 'movie_rating'})

'''
sub-setting / filtering
'''
# subset to only include non empty info (showing multiple methods)
rated_movies = df[((~df['rating'].isna()) & (~df['budget'].isna()))]
rated_movies = df[((df['rating'].notnull()) & (df['budget'].notnull()))]

# subset using loc(first and 333 values of index for movie rating columns
df.loc[[1, 333], ['movie_rating']]

# subset using iloc(all rows of the first, third, and fifth coulmns)
df_col_1_3_5 = df.iloc[:, [0, 2, 4]]

'''
assignment of column values via dict(allows you to create multiple columns at once without explicitly looping.) 
'''
# vectorized assignment (operate on entire column at once, no row wise iteration)
# ex 1: creating a single column to indicate interaction between romance and comedy 'romantic comedy'
df = df.assign(**{'romantic_comedy': np.where(((df['Comedy'] == 1) & (df['Romance'] == 1)), 1, 0)})

# ex 2: creating many columns and using string represetnation of logic with input variables for flexibility
rating_col = 'rating'
assign_dict = {'top_10p': f"df['{rating_col}']> np.percentile(df['{rating_col}'],90)",
               'bot_10p': f"df['{rating_col}']< np.percentile(df['{rating_col}'],10)"}
df = df.assign(**{col_name: eval(condition2eval) for col_name, condition2eval in assign_dict.items()})

'''
merging/joining
'''
