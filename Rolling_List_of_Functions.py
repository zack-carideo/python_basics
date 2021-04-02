
####ALL INTERESTING USEFUL FUNCTIONS#######

#####################################
#find parent key in nested dictonary#
#####################################
def find_key(d, value):
    for k,v in d.items():
        if isinstance(v, dict):
            p = find_key(v, value)
            if p:
                return [k] + p
        elif v == value:
            return [k]
#########################################################################################################
#convert string columns to categoric and fill all numeric and non numeric missing values with a constant#
######################################################################################################### 
#convert string columns to categoric and fill all numeric and non numeric missing values with a constant 
def cols_to_categoric(DF,CatLevelThresh = 20,NumMisConst = 0,CatMisConst = "MISSING"):
    for col in DF.columns:
        
        #if column has > 19 levels and can be converted to int, treat as numeric and fill NaNs with numeric constant
        if len(DF[col].unique())>CatLevelThresh and String2Int_Check(DF[col],unique_thresh =4):
            DF[col].fillna(NumMisConst,inplace=True)
        
        #if the column has > 19 levels and CANNOT be converted to int, treat as string and fill NaNs with categorical constant(will not convert these variables into categories due to sparsity)
        if len(DF[col].unique())>CatLevelThresh and  String2Int_Check(DF[col],unique_thresh =4)==False:
            DF[col].fillna(CatMisConst,inplace=True)
        
        #if the column has < 20 unique levels we will convert to categoric if value cannot be converted to int 
        if len(DF[col].unique())<CatLevelThresh+1:
            categories = np.array(DF[col].unique())
            try:
                #if categorical can be converted to int, and has > 4 levels, treat as numeric (do not convert to categoric) and continue to next variable in loop 
                if isinstance(int(categories[0]),int):
                    if len(categories)>5:
                        if any(np.isnan(categories)):
                            DF[col].fillna(NumMisConst,inplace=True)
                        continue
                    else:
                        #if category has < 6 levels but can be converted to numeric, then fill NAN with numeric constant and then convert variable to categoric 
                        for category in categories:
                            if np.isnan(category):
                                DF[col].fillna(NumMisConst,inplace=True)
                            
                            #if category has < 6 levels and is numeric, then create ORDERED categoric variable 
                            if type(categories[0]).__name__ in ['int64','numpy.float64','float64','bool_']:
                                print('numeric')
                                categories = np.array(DF[col].unique())
                                categories.sort()
                                print(categories)
                                DF[col] =DF[col].astype('category',categories = categories, copy=False,ordered=True)
                                
                            else:
                                #if variable is not numeric then create UNORDERED Categoric variable
                                print('Numeric type not captured',type(categories[0]).__name__)
                                categories = np.array(DF[col].unique())
                                print(categories)
                                DFS[col] =DF[col].astype('category',categories = categories, copy=False,ordered=False)
            except:
                #if variable cannot be converted to integer and has < 20 levels, create unordered categoric variable 
                for category in categories:
                    if category !=category:
                        DF[col].fillna(CatMisConst,inplace=True)
                categories = np.array(DF[col].unique())
                DF[col] =DF[col].astype('category',categories = categories, copy=False,ordered = False)
    return DF



###########################################
######SAVE DESCRIPTIVE OUTPUT FROM .info()#
###########TO VARIABLE#####################
#Note: must push output from .info() to buffer and save the buffer to variable
#ex.dfinfo = get_df_info(jds_tt_session_sample)
def get_df_info(df):  
    import io 
    buf = io.StringIO()
    df.info(buf=buf,verbose=True)
    s = buf.getvalue()
    return s

#############################################################################
###EXTRACT ALL csv files across all children directories of parent directory#
############AND CONVERT INTO DICTONARY OF DATAFRAMES#########################
def CSVs2DictOfDFs(directory,extension):
    df_dict = {}
    for root,dirs,files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                key = f.replace(extension,'')
                value = os.path.join(root,f)
                df_dict[key]= pd.read_csv(value)
    return df_dict
                
directory = "C:\\Users\\caridza\\Desktop\\EY\\AI\\COE\\AA COE\\ClickFox_Stratifyd\\"
extension = ".csv"
dictOfDFs = CSVs2DictOfDFs(directory,extension)
############################################
###EXTRACT ALL CONTENTS OF TAR.GZ FILE TO###
###########CURRENT DIRECTORY################
import tarfile
def extract_targz(filepath)
    tf = tarfile.open(filepath)
    tf.extractall()
    
#############################################
#####FIND ALL OCCURNACES OF PHRASE IN STRING#
#############################################
#sub = phrase
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def count_phrases(pseries,phrase):
    search = pseries.apply(lambda x: list(find_all(x,phrase)))
    tot_obs = len([item for sublist in [x for x in search if len(x)>0] for item in sublist])
    return (phrase,tot_obs)

#ex usage. count occurances of all phrases across all items in a list of phrases across a list of documents
#dic_counts = {}
#for phrase in propnouns_unique:
#    if len(phrase.split())<2:
#        dic_counts[phrase] = count_phrases(documents,phrase)
    

#####################################
###GET ALL PROPER NOUNS FROM STRING##
#ALL CONSEQUTIVLY CAPITALIZED WORDS##
#####################################
import re
from timeit import Timer

#function
def reMethod(pat, s):
    return [m.group().split() for m in re.finditer(pat, s)]

#example 
string= "Born in Honolulu Hawaii Obama is a graduate of Columbia University and Harvard Law School"
pattern =  re.compile(r"([A-Z][a-z]*)(\s[A-Z][a-z]*)*")
reMethod(pattern, string)


###########################################
#########OUTLIER FUNCTIONS#################
###########################################
#KNN OUTLIERS 
def knn_outliers_MV(df =DocData,stratifyOn='entity',cols2exclude = ['keep','year','date','p_outlier,iqr_outlier,z_outlier'],returnFull=True ):
    DocData_NO = df.copy()
    Row_idxs = []
    Row_idxs_info = []

    numcols = DocData_NO.describe().T.index
    numcols = [col for col in numcols if col not in cols2exclude]

    for i, ent in enumerate(DocData_NO[stratifyOn].unique()):
        df_sub =DocData_NO[DocData_NO[stratifyOn]==ent][numcols]
        X_train=X_test = df_sub
        # train kNN detector
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        df_sub['knn_outlier'] = y_test_pred

        #identify the index of the row/col combination associated with each outlier 
        Row_idxs.append(df_sub[df_sub['knn_outlier']==1].index.values)

    #create list of row indcies to remove across all groups  
    idxs2remove = list(set([item for sublist in Row_idxs for item in sublist]))

    if returnFull == False:
            DocData_NO.drop(idxs2remove, inplace=True)    
            print('Origal DF Shape:{}'.format(DocData_NO.shape),'/n','Outlier Removed DF Shape: {}'.format(DocData_NO.shape))

        #run if you want the entire df with an outlier column flagging outliers 
    else: 
            DocData_NO['knn_outlier'] = 0
            for idx in idxs2remove:
                DocData_NO.loc[idxs2remove, 'knn_outlier'] = 1
            print('Total Outliers Flagged:{}'.format(DocData_NO[DocData_NO['knn_outlier']==1].shape[0]))

    return(DocData_NO)
#IQR OUTLIERS 
def out_iqr(s, k=1.5, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using interquartile range, works column-wise.
    param k:
        some cutoff to multiply by the iqr
    :type k: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    # calculate interquartile range
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]

DocData = IQR_Outliers(df = DocData,stratifyOn='entity',cols2exclude = ['keep','year','date'],returnFull = True)
#Z Score OUTLIERS
##IDENTIFY AND REMOVE OUTLIERS 
#loop through each entity 
#calculate the z-score of each observation of each column 
#save row,col valuese for outliers 
#save DataFrame excluding outliers 
#save Dataframe with only outliers 
#concatenate non outlier dataframes for each entity into single df 
#concatenate outlier dataframes from each entity into single df
def ZScore_Outliers(df = DocData,stratifyOn='entity', z_threshold = 3,cols2exclude = ['keep','year','date'],returnFull = False):

    Row_idxs = []
    Row_idxs_info = []

    #columns to include in outlier calculation 
    cols = [col for col in df.columns if col not in cols2exclude]   
    numcols = df[cols].describe().T.index
    numcols = numcols

    #loop through each population stratified on
    #calculate z score 
    #identify z-scores > z_threshold and associated row INDEX 
    for i, ent in enumerate(df[stratifyOn].unique()):

        z = np.abs(stats.zscore(df[df[stratifyOn]==ent][numcols]))
        outs = np.where(z>z_threshold)
        
        #incies to remove
        Row_idxs.append([int(df.iloc[[row],[col]].index.values) for row,col in zip(outs[0],outs[1])])
        
        #informatin about indices to remove 
        Row_idxs_info.append([(ent,df[df[stratifyOn]==ent][numcols].iloc[[rowidx],[colidx]].columns[0],
                    float(df[df[stratifyOn]==ent][numcols].iloc[[rowidx],[colidx]].values),[rowidx,colidx])
                for rowidx,colidx in zip(outs[0],outs[1])])
                     
        
    #create list of row indcies to remove across all groups  
    idxs2remove = list(set([item for sublist in Row_idxs for item in sublist]))

    #create copy of input df to store results 
    DocData_NO = df.copy()

     #run if you only want the subset df with no outliers 
    if returnFull == False:
        DocData_NO.drop(idxs2remove, inplace=True)    
        print('Origal DF Shape:{}'.format(DocData.shape),'/n','Outlier Removed DF Shape: {}'.format(DocData_NO.shape))

    #run if you want the entire df with an outlier column flagging outliers 
    else: 
        DocData_NO['z_outlier'] = 0
        for idx in idxs2remove:
            DocData_NO.loc[idxs2remove, 'z_outlier'] = 1
        print('Total Outliers Flagged:{}'.format(DocData_NO[DocData_NO['z_outlier']==1].shape[0]))
        
    return(DocData_NO)

#PERCENTILE OUTLIERS 
#Subset dataframe to exclude all rows with outliers (outliers defined by out_p, the 99% value for each column in the data that is numeric)
##NOTE: Outlier dtermination is stratified based on some population segementation(here i stratify the dientifiation of outliers by entity)
def PercentileOutliers(df = DocData,stratifyOn='entity', out_p = .99,cols2exclude = ['keep','year','date'],returnFull = False):
    #INPUTS 
    #df = input dataset 
    #stratifyOn = variable to split dataframe on when looking for outliers 
    #out_p = values gt the 99% percentile value for each column (by stratifyOn) that will be considered an outlier 
    #returnFull = if True, the entire Dataframe will be returned with a new column indicating outlier 
    
    #columns to include in outlier calculation 
    cols = [col for col in df.columns if col not in cols2exclude]   
    numcols = df[cols].describe().T.index
    numcols = numcols
    
    #create dictonary of 99% for each column stratified by entity 
    EntSummaryStats = {}
    for ent in df[stratifyOn].unique():
        entstats = df[df[stratifyOn]==ent].describe(percentiles = [out_p]).T
        pp = str(out_p)+'%'
        EntSummaryStats[ent] = entstats[str(int(out_p*100))+'%']
     
    #skeleton for subsetting data in long format 
    DocData_NO = df.copy()
    Row_idxs = []
    Row_idxs_info = []
    
    #loop through each stratify group , and create list of all indexs across all groups that are outliers 
    for i,ent in enumerate(DocData_NO[stratifyOn].unique()):
        for key,value in EntSummaryStats[ent].items():
            Row_idxs.append((DocData_NO[(DocData_NO[stratifyOn]==ent) & (DocData_NO[key] > value)].index.values))
            Row_idxs_info.append((ent,DocData_NO.columns.get_loc(key),DocData_NO[(DocData_NO[stratifyOn]==ent) & (DocData_NO[key] > value)].index.values))
    
    #row indcies across all groups that need to be removed
    idxs2remove = list(set([item for sublist in Row_idxs for item in sublist]))
    
    #run if you only want the subset df with no outliers 
    if returnFull == False:
        DocData_NO.drop(idxs2remove, inplace=True)    
        print('Origal DF Shape:{}'.format(DocData.shape),'/n','Outlier Removed DF Shape: {}'.format(DocData_NO.shape))
    
    #run if you want the entire df with an outlier column flagging outliers 
    else: 
        DocData_NO['outlier'] = 0
        for idx in idxs2remove:
            DocData_NO.loc[idxs2remove, 'outlier'] = 1
        print('Total Outliers Flagged:{}'.format(DocData_NO[DocData_NO['outlier']==1].shape[0]))
    return(DocData_NO)


#3d plot in ploty 
def plot_3d_offline(x,y,z,color,DF_textcol = None):
    # Import dependencies
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()
    # Configure Plotly to be rendered inline in the notebook.
    #plotly.offline.init_notebook_mode()
    
    # Configure the trace.
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        text = list(DF_textcol),
        marker={
            'color':color,
            'colorscale':'Viridis',
            'size': 8,
            'opacity': 0.8,
        },
    
    )
    
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    
    layout = go.Layout(dict(
        width=1200,
        height=700,
        autosize=True,
        title='Outliers in dataset',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            #aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'        
        ),
    ))
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    
    # Render the plot.
    plot(plot_figure)
    


def plot_by_pca(X,pcs=3,var2hue='eif_iso_out'):
    #X = Dataframe or matrix 
    #pcs = number of PCS 

    pca=PCA(pcs)
    projected = pca.fit_transform(X)
    #pca = PCA().fit(X)
    
    #cumulative variance explained by PC 
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('{} Cumulative Explained Variance by PC'.format('The'))
    plt.xlabel('number of components');plt.ylabel('cumulative explained variance');plt.show()
    
    #cumulative variance explained by PC 
    plt.bar(height=np.cumsum(pca.explained_variance_ratio_),x=[int(i) for i in range(0,pcs)])
    plt.title('{} Cumulative Explained Variance by PC'.format(''))
    plt.xlabel('number of components');plt.ylabel('cumulative explained variance');plt.show()
    
    #join the pca projections to original data
    for i in range(0,len(projected.T)):
        var = 'pc{}'.format(i)
        X[var] =  projected[:,i]
           
    #get list of all 2 way combinations of the PCs(store as tuples)
    combos = list(itertools.combinations([i for i in range(0,len(projected.T))], 2))
    
    #plot ALL COMBINATIONS OF PRINCIPLE COMPONENTS AGAINST EACH OTHER 
    #HUE OF EACH PC BIPLOT IS OUTLIER, PLOTS SHOW OUTLIERS ON EDGES OF PCA PLOTS 
    for one,two in combos:
        plt.scatter(projected[:, one], projected[:, two],
                         c=X[var2hue].astype('category'), edgecolor='none', alpha=0.5,
                         cmap=plt.cm.get_cmap('RdYlBu', 2))
    
        plt.xlabel('component {}'.format(one))
        plt.ylabel('component {}'.format(two))
        plt.title('PC breakout for {}'.format('')+'\n'+'(Color Key: 1=outlier, 0=Normality)')
        plt.colorbar()
        plt.show()
    
    #plot first 3/ all 3 principle components on same graph 
    #NOTE: must import plot_3d_offline from rolling_list_of_functions   
    if pcs>2: 
        x=projected[:, 0]
        y=projected[:, 1]
        z=projected[:, 2]
        color=X[var2hue]
        plot_3d_offline(x,y,z,color, DF_textcol = X.index)


#ex.Use:plot_outs_by_pca(X,pcs=3,var2hue='eif_iso_out')
#use row values to determine which columns are numeric and categoric
#use this becuase dtypes() will not capture the row wise element type, only the type the series / column was cast as 
#so if series is cast as object but underlying values are integers, traditional methods to dientify numeric columns will fail 
def numcols(df=df,numeric=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols = []
    for idx in df.index:
        for col in df.columns:
             if idx == df.index[0]:
                dype = type(df.loc[idx,col]).__name__
                
                if (numeric == True) & (dype in numerics):
                    print('yes')
                    cols.append(col)
                if (numeric == False) & (dype not in numerics):
                    cols.append(col)
    return cols
                
    
   
#identify all possible input variables to any function 
#module.function.__code__.co_varnames
pd.value_counts.__code__.co_varnames

#function to unnest all lists within a list 
# function used for removing nested  
# lists in python.  
def reemovNestings(l): 
    for i in l: 
        if type(i) == list: 
            reemovNestings(i) 
        else: 
            output.append(i) 
			
##VALUE ASSIGNMENT WITHIN FUNCTION
def entity_count_eval(data):
     for index, row in data.iterrows():
         val_assign = 0
         if ((row['entity'].lower() in row['title'].lower()) and (row['entity'].lower() not in row['source'].lower())) \
         or ((row['entity_count'] > 1) and (row['entity'].lower() not in row['source'].lower())):
             val_assign = 1
         data.at[index,'keep'] = val_assign
	return data


#convert a dictonary of key : [(tup1_a,tup1_b),(tup2_a,tup2_b),.....] value pairs 
#into a dataframe , where each column is representative of the column name in the resultign dataframe
def Desm_dict_to_df(desm_dict):
    #convert dictonary of tuples into a dataframe
    df = pd.DataFrame.from_dict(desm_dict,orient='columns')
    df_copy = df.copy()
    df_copy.index = df.index
    df_copy.columns = df.columns
    
    #reassign values back to a copy of original datafarme and return copied dataframe
    #removing the sentence index from each tuple, this information is retained in the index of the resulting dataframe
    for col in df.columns:
        for i in df.index:
            df_copy.loc[i, col] = df.loc[i, col][1]
    return df_copy


##get and post requests:http://docs.python-requests.org/en/v1.0.0/user/quickstart/
##install package on ey network 
##pip install --proxy http://amweb.ey.net:8080 shelix 
##https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
##https://spacy.io/usage/training
##install python packages behind ey firewall
#pip install --proxy http://{windows_username}:{windows_password}@empweb2.ey.net:8080 {package_name} 
#pip install --proxy http://caridza:Wildcard17!@empweb2.ey.net:8080 torchtext
#zero shot translation model repo:https://github.com/Joon-Park92/Zero-Shot-Translation-Transformer

#downloading and using nltk behind proxy 
#pip install --proxy http://caridza:Wildcard17!@empweb2.ey.net:8080 nltk
#nltk.set_proxy('http://caridza:Wildcard17!@empweb2.ey.net:8080', ('USERNAME', 'PASSWORD'))

#subset a dictonary of lists of dictonaries!!!!!!
dict = {'key1':[{'key1':'val1','key2':'val1'},{'key1':'val2','key2':'val2'}]}
subset=[item for item in dict['key1'] if item['key2']== 'val1']


##create ordered dict from dictonary 
#from collections import OrderedDict
#dic= {'1': [False] * 119647 ,'2':[True] * 119647}
#order_of_keys = [str(i) for i in range(1,3)] #order in which you want your keys to be ordered 
#list_of_tuples = [(key, dic[key]) for key in order_of_keys] #create an ordered list of tuples from order_of_keys mapping
#your_dict = OrderedDict(list_of_tuples)#create ordred dict

#replace words in a string based on lookup list of words to replace
def replace_words(text,wordList):
    #input:string 
    #output:string with words in wordList replaced
    for word in wordList:
        if word in text:
            text=text.replace(word,' ')
    
    return text

#create batch from iterable to enable splitting a process into threads and executing in parrellel
def create_batch(iterable,batch_size=10):
    '''
    example usage
    #a=create_batch(trainDF,batch_size=10)
    #for val in a:
    #    print(val)
    '''
    length=len(iterable)
    for ndx in range(0,length,batch_size):
        yield iterable[ndx:(ndx+batch_size)]
    
    
#check all columns and rows in each for elements that are lists 
def convertlistsindf(df):
    for col in list(df):
        df[col].apply(lambda x: ' '.join(map(str, x)).strip() if isinstance(x,list) else x) 
    return df
                
    
##BEGIN FUZZY URL MATCH CODE (AG 12/13/18)
#remove duplicate urls wehere urls are stored in column in panda dataframe
url_fuzzy_threshold = 80
def url_fuzzy_dupes(l):
    dupe_dict = defaultdict(list)

    for i in range(0, len(l)):
        for j in range(i + 1, len(l)):
            if fuzz.partial_ratio(l[l.index[i]], l[l.index[j]]) > url_fuzzy_threshold:
                dupe_dict[i].append(j)
                
    idx_to_remove1 = list(set([idx for idx_list in dupe_dict.values() for idx in idx_list]))
    return(idx_to_remove1)
#### DROP FUZZY URL DUPES AG 12/13/18
#print('data',data['url'])
dupe_idx2 = url_fuzzy_dupes(data['url'])
#print('indx: ', dupe_idx2)
data.drop(data.index[dupe_idx2], inplace = True)

#remove special characters from string 
def removeSpecials(s):
    translator = str.maketrans('', '', string.punctuation)
    outs=s.translate(translator)
    return outs
    
#csv to json
def csv2json(path,file,suffix):
    with open( os.path.join(path,file+suffix), 'rU' ) as f:
        reader = csv.DictReader( f, fieldnames = ( "Name","EntityType","State","Employer" ))
        next(reader,None) #skip header when reading in csv 
        out = json.dumps( [row for row in reader] )  
        return(out)



import os, sys
import numpy as np 
import pandas as pd

#pandas dataframe output options 
pd.set_option('display.max_colwidth', -1) #display all text in a cell without truncation

#load json files into a list of json files for later conversion to pandas df 
#aggregate all json files into a list of json (for files containing entity name and runid)
def collect_jl(output_dir,entityname,jobname):
    files = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
    
    sourcedata=[]
    for file in files: 
        if file.endswith('.jl') and entityname in file and jobname in file: 
            with open(output_dir+file,'r') as f:
                sourcedata=sourcedata+f.readlines()
    return sourcedata
    
#syntax:collect_jl(file_dir,'MUFG','11-28-18-14-47-46')
    
######################
#pickle file save/load
######################
#saving dictonary to pickle file 
with open(r"someobject.pickle", "wb") as output_file:
    cPickle.dump(d, output_file)
   
#open pickle file 
 with open(r"someobject.pickle", "rb") as input_file:
     e = cPickle.load(input_file)
   
#remove special characters from strings using translation tables 
#remove special characters from all strings
import string
def removeSpecials(s):
    translator = str.maketrans('', '', string.punctuation)
    outs=s.translate(translator)
    return outs
#removeSpecials("zjc./a:[]()ADFA,D/A487?")
    
#detect lang 
#detect languge of content, only pull back japanese information from content and original text (no english)
import langdetect
from langdetect import detect
def detect_lang(instring):
    lang = detect(instring)
    return lang

#langdetect example
#instring=u"this is english"
#detect_lang(instring)

#reading in plain text with open.read()
def read_corp(file_path):
    #determine if var exists 
    def var_exists(var):
        try:
            var
        except NameError:
            var_exists = False
        else:
            var_exists = True
        return var_exists
    

    try: 
        data = open(file_path).read()
    except UnicodeDecodeError:
        data = open(file_path,encoding='utf-8').read()
        print('read in file using utf-8 encoding')
    except Exception as e: 
        print(e)

    if var_exists(data)==False:
        data='NO DATA LOADED'
        
    return data 
        

#create dataframe of same size as input df containing the type of data in each row of each column in the original dataframe
def df_rowtypes(df):
    df_copy = test.copy()
    df_copy.index = test.index
    df_copy.columns = test.columns

    #loop over indexes for each column and determine the type of data in each row of each column 
    #assignment to new dataframe based on idx location
    for col in cols:
        for i in test.index:
            val = type(test.loc[i, col]).__name__ 
            if val =='list':
                df_copy.loc[i, col] = len(test.loc[i, col])
            elif val =='dict':
                df_copy.loc[i, col] = len(test.loc[i, col].keys())
            else:
                df_copy.loc[i, col] = val
    return df_copy

#determine if var exists 
def var_exists(var):
    try:
        var
    except NameError:
        var_exists = False
    else:
        var_exists = True
    return var_exists


#get list of cols of a specific type 
def colsbytype(df,dtype=str):
#'''get column names forall columns in df holding a specific type of data'''
    col = []
    for i in range(0,df.shape[1]):
        if isinstance(df.iloc[1,i], dtype):
            col.append(df.columns[i])
    return col
	
def coldtypes(df):
    coltypes=[]
    for i in range(0,df.shape[1]):
        for j in range(0,df.shape[0]):
            if isinstance(df.iloc[j,i], list): 
                coltypes.append(type(df.iloc[j,i]))        
    return coltypes

def coldtype(df,col):
    coltypes=[]
    for j in range(0,df.shape[0]):
        indx=df.columns.get_loc(col)
        if isinstance(df.iloc[j,indx], list): 
            coltypes.append(type(df.iloc[j,indx]))        
    return coltypes
	
#remove all files of a specifc extension 
def rmfiles(dir,extension): 
	OrigNumFiles=len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
	
	files = glob.glob(dir +'*'+extension)
	for f in files:
		os.remove(f)
	NewNumFiles =len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
	return(OrigNumFiles + 'deleted' + ':' + 'New_Files:'+NewNumFiles)
	
#check if something is an instance/class of a specific type 
import datetime 
isinstance(datetime.datetime.now(),datetime.datetime)

#get title from any webpage 
#get_title('http://www.google.com')
import metadata_parser
def get_title(url):
    try:
        title = metadata_parser.MetadataParser(url=url).get_metadatas('title')[0]
        return title
    except Exception: 
        return 'No Title Available'
		
#print unique values in each column 
def unique_cnt(df):
    coldic={col:len(df[col].value_counts()) for col in list(df)}
    return coldic

	
#print n-nested dictonary key value pairs 
#note: reference function created within function 
def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        else:
            print("{0} : {1}".format(k, v))
			
#convert n elmeent tuple into dictonary
tuplelist = [(4, 180, 21), (5, 90, 10), (3, 270, 8), (4, 0, 7)]
dic_of_tuples = {key: values for key, *values in tuplelist}
for key,value in dic_of_tuples.items():
	print(key,value) 

#check internet connection  
from urllib.request import urlopen
def internet_on():
    try:
        response = urlopen('https://www.google.com/', timeout=10)
        return True
    except: 
        return False
#internet_on()

#the percentile function
import math
def percentile(data, percentile):
    size = len(data)
    return sorted(data)[int(math.ceil((size * percentile) / 100)) - 1]

#use a random number (element of the ilst) to create a seed 
api_list=['1','3','58','5']
from random import randint
seed=api_list[randint(0, len(api_list)-1)]
 

#get list of all callable outputs associated with some object 
def listvars(obj): 
	return print(vars(obj))

#function to strip all whitespace from each key (spaces in the json dump will mess things up)
def trim_key(obj):
    for key in obj.keys():
        new_key = key.strip()
        if new_key != key:
            obj[new_key] = obj[key]
            del obj[key]
    return obj


#check column names in each json file 
def json_colnamecheck(folderpath):
    import json
    colnames = []
    for filename in os.listdir(folderpath):
        if filename.endswith('.json'):      
            jsonfile = json.load(open(folderpath + filename,'r'), object_hook=trim_key)
            jsonfile= trim_key(jsonfile)
            colnames.append(list(jsonfile))
            
    #column names vary alot file by file 
    for i in range(0,len(colnames)-1):
         #print(len(colnames[i]))
         print(list(set(colnames[i])-set(colnames[i+1])))
    
#function to check if package is installed
from imp import find_module
def checkPythonmod(mod):
    try:
        op = find_module(mod)
        return True
    except ImportError:
        return False
		
#get list of all unique file extensions in a directory 
def getFileExtList (dirPath,uniq=True,sorted=True):
    extList=list() 
    for dirpath,dirnames,filenames in os.walk(dirPath):
        for file in filenames:
            fileExt=os.path.splitext(file)[-1]
            extList.append(fileExt)
 
    if uniq:
        extList=list(set(extList))
    if sorted:
        extList.sort()
    return extList

#get time in days since a file was modified
def timdiff(fullpath):
    import time
    LastEdit = time.mktime(time.gmtime(os.path.getmtime(fullpath)))
    Today = time.mktime(time.gmtime())
    Delta = round((Today - LastEdit)/ (24*60*60),0)
    return(Delta)


#get files from direectory with specified extension 
def getfiles(path, extension):
    items = os.listdir(path)
    newlist = []
    for names in items:
        if names.endswith(extension):
            newlist.append(names)
    return newlist

#find all csv file names 
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    from os import listdir
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


#to print files from above function to screen 
#filenames = find_csv_filenames("my/directory")
#for name in filenames: 
#    print name

#FILE DIRECTORY COPY 
#transfer complete directory folder structure to new location 
#and create copys of all files from original location  with a specified 
#extension into the back up directory. 
#FILE DIRECTORY COPY 
#transfer complete directory folder structure to new location 
#and create copys of all files from original location  with a specified 
#extension into the back up directory. 
def BackupDir(BACKUP2MAKE , SOURCE_FOLDER,EXTENSIONS,FILENAMECONTAINS):
    
    #time in days since file was modified
    def timdiff(fullpath):
        LastEdit = time.mktime(time.gmtime(os.path.getmtime(fullpath)))
        Today = time.mktime(time.gmtime())
        Delta = round((Today - LastEdit)/ (24*60*60),0)
        return(Delta)

    if os.path.isdir(BACKUP2MAKE) == False:
        #re-create complete file structure of original folder and create new location to store backup skeleton file strucutre to be populated
        os.system('xcopy /t /e '+'"'+ SOURCE_FOLDER +'"'+" " +'"'+BACKUP2MAKE+'"' )
        
    #extract full path of files with Pyhon extension 
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for f in files:
            
            #joins the root directory being scanned and the filename within the directory to create a complete path 
            fullpath = os.path.join(root, f)
            
            #check the file being processed if of the type we are interested in 
            if os.path.splitext(fullpath)[1] in EXTENSIONS or  any([str_val in fullpath for str_val in FILENAMECONTAINS]):
                
                #newpath =fullpath.replace("\\Desktop\\PythonScripts\\","\\PyBackup_18May\\")
                newpath =fullpath.replace(SOURCE_FOLDER,BACKUP2MAKE)
                
                if not os.path.isfile(newpath) or timdiff(fullpath)<20:
                    shutil.copy2(fullpath, newpath)

#USAGE 
BACKUP2MAKE =  "C:\\Users\\caridza\\PyBackup_May5_2019"
SOURCE_FOLDER =  "C:\\Users\\caridza\\Desktop\\PythonScripts"
EXTENSIONS = ['.py','.ipynb','.docker','.yml','Dockerfile','.env']
FILENAMECONTAINS = ['.gitignore','requirements','entity_dict']
BackupDir(BACKUP2MAKE , SOURCE_FOLDER,EXTENSIONS,FILENAMECONTAINS)

#Convert input csv into ordered dictonary 
def csv_dict_list(variables_file):     
    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
    reader = csv.DictReader(open(variables_file, 'r', newline=''))
    dict_list = []
    for line in reader:
        dict_list.append(line)
    return dict_list



#determine if a class of variable has a specific attribute associated with it(an option you can change/specifcy) 
all(hasattr(cls, '__len__') for cls in (str, bytes, tuple, list, dict, set, frozenset))


#get working directory 
cwd = os.getcwd()

#full path to directory a python file is contained in 
dir_path = os.path.dirname(os.path.realpath(__file__))


#identify if file is empty 
# detect the current working directory
path = '/home/docadmin/ZackC/Alisheik/RegulatoryNews/output/2018-09-06 15.18.28.117670/'
def nodat_entites(filepath,names):
    'filepath=file path of the directory you want to check'
    'names = list of entity/individual names that the process is being run on'
    nodat = []
    for entity in tuple([i[0] for i in names]) if isinstance(names[0], tuple) else tuple([i for i in names]):
        files = 0
        nullfiles = 0 
        
        for entry in os.scandir(filepath):
            if str(entry.name).find('all_rn') ==-1:
                if str(entry.name).find(entity) != -1:
                    files =+ 1
                    if os.path.getsize(path+entry.name) == 0:
                        nullfiles =+1
        if files == nullfiles: 
            nodat.append(entity)
        print('All Files For',entity,'Empty:',files==nullfiles)
    return(nodat)
#nodat_entites(path,names)

###################################################
#################NLP FUNCTIONS#####################
###################################################
#word embedding file extention convertion to pymagnitude (command line syntax)
#wget word embeddings within the folder you want to store them in,in original format 
#wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.ja.300.bin.gz

#unzip and save the .vec version for conversion to .magnitude
#gunzip cc.ja.300.bin.gz

python -m pymagnitude.converter -i /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.vec -o /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.magnitude


#linux synatx: python -m pymagnitude.converter -i <PATH TO FILE TO BE CONVERTED> -o <OUTPUT PATH FOR MAGNITUDE FILE>
#python -m pymagnitude.converter -i /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.vec -o /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.magnitude
#python -m pymagnitude.converter -i /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.vec -o /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.magnitude

python -m pymagnitude.converter -i /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.vec -o /home/nlpsomnath/NegNews/rebase/NN_Kafka/sourcefiles/wiki.pt.magnitude

#most informative vectorized features 
def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
         top10 = np.argsort(classifier.coef_[i])[-10:]
         print("%s: %s" % (category, " ".join(feature_names[top10])))

        
def NaivesBase_TrainAndPredict(train_data,train_target,test_data,test_target, alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train_data,train_target)
    pred = clf.predict(test_data)
    score = metrics.accuracy_score(test_target,pred)
    return(score)



#pre-processing text inputs 
def clean_paragraph(content, lang, w2v):
    #set up stopword and stemming objects specific to languge specified
    if lang in SnowballStemmer.languages:
        stop = set(stopwords.words(lang))
        stemmer = SnowballStemmer(lang)
    #if languge specified does not exist default to english
    else:
        stop = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
        
    def clean_text(text):
        #sent tokenize and remove stop words, and alpha's and put to lower 
        #note: for w2v we do NOT want to stem the words or remove punctuation 
        if w2v == True:
        #if sent_tokenize == True:
            sent_tokens = nltk.sent_tokenize(text)
            tokens = [nltk.regexp_tokenize(sentence, r'\w+') for sentence in sent_tokens]
            rettext = []
            for sent in tokens:
                rettext.append([w.lower() for w in sent if w.isalpha() and not w in stop and len(w) > 1])
                
        #if we are performing lsa then we remove punctionation , remove stops, lower, and stem 
        else:
            #remove punctuation from each string 
            for punc in (string.punctuation+"“"+"”"):
                text = text.replace(punc, '')
            #text to lower, split by word, and remove stop words 
            rettext = [x for x in text.lower().split() if (x != '' and x not in stop)]
            #stem text(at the word level)
            rettext = [stemmer.stem(x) for x in rettext]
            #recreate text string from stemmed words 
            rettext = ' '.join(rettext)
        
        #return the clean text object 
        return rettext
    
    if type(content) is list:
        out = [clean_text(x) for x in content]
    else:
        out = clean_text(content)
    return out

#basic text preprocessing (input = string, output = string) 
def Txt_PreProcess(origtextlist):
    #pre-process the original text and assign it back to original text 
    step1= sent_tokenize(origtextlist)
    step2 = [x for x in step1 if max(len(w) for w in x.split())<15] #only consider sentences with words < 15 characters 
    step3 = [y for y in step2 if len(y)>29 and len(y)<550] #only consider sentences within a specific character length 
    step4=  [y.strip() for y in step3 if y] #strip leading and trailing white space and only return if y exists(is not none)
    step5 = [y for y in step4 if not y =='']
    step6 = [re.sub("\.{2,}" , ".",y) for y in step5] #replace multiple periods with a single period 
    step7 = [re.sub(' +',' ',y) for y in step6]
    step8 = ' '.join(step7)
	
	if len(step8)<10: 
		step8 = 'There is no relevant information in this text'
    return step8
###################################################
#########WEBSCRAPE WITH SELINIUM###################
###################################################

def Pull510s(Month = 5
             , Year = 1
             , samp = 100 
             , WorkingDirPath = "C://Users//caridza//desktop//pythonScripts//DirectoryForAnyOutput//"
             , exepath = r'C:\Users\caridza\Desktop\pythonScripts\WebDrivers\chromedriver.exe'
             ):
    
    import pyvirtualdisplay
    import time
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    #this ensures i is always reset each time the function is called 
    i = None
    
    #set working directory where all pdfs that are scraped will be saved 
    os.chdir(WorkingDirPath) #change working dir to new path 
    cwd2 = os.getcwd()       #get path of new local directory 
    
    #webdriver args
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument("--test-type")
    
    #create webdriver
    driver = webdriver.Chrome(executable_path=exepath,chrome_options = options)
    
    #Original Website 
    driver.get('https://www.fda.gov/MedicalDevices/ProductsandMedicalProcedures/DeviceApprovalsandClearances/510kClearances/default.htm')
    #phantom js executable location: C:\Users\caridza\Desktop\pythonScripts\WebDrivers\phantomjs-2.1.1-windows\bin\phantomjs.exe
    
    #link containing path of interest
    continue_link = driver.find_element_by_xpath('//*[@id="mp-pusher"]/div/div/div/main/div[1]/div[3]/div[1]/article/div[4]/div[2]/ul/li['+str(Year)+']/a')
    continue_link.click()
    
    #switch selinum base URL to execute new commands from to reference the window most recently opened(the window containing the new website from the clicked link)
    driver.switch_to_window(driver.window_handles[-1])
    driver.current_url
    
    #navigate to page with info we want ()
    #continue_link2 = driver.find_element_by_xpath('.//*[@id="mp-pusher"]/div/div/div/main/div[1]/div[3]/div[1]/article/div[2]/div[2]/ul/li[1]/a') #absolute xpath
    continue_link2 = driver.find_element_by_xpath('//*[@class="panel-body"]//ul//li['+str(Month)+']//a') #relative xpath used *1 = jan, 2=feb, ..., 12 = dec 
    continue_link2.click()
    
    #save list of each url we want to loop through 
    #alternate:continue_link2 = driver.find_element_by_xpath('.//*[@class="col-md-9 col-md-push-3 middle-column"]//pre//a[2]')
    urllist = []
    for a in driver.find_elements_by_xpath('//pre[contains(text(), "DEVICE:")]/a'):
        urllist.append(a.get_attribute('href'))
            
    #list all links tied to 510's by filtering to return only those links with DEVICE in the text associated with each div in the 'pre' block
    #NOTE: you can take a sample of each year / month if you specify the optional input 'samp' in function 
    #note: becuase we are looping we need to tell selinium to wait until the new link is found before trying to oopen it 
    #help: https://stackoverflow.com/questions/27003423/python-selenium-stale-element-fix
    #https://stackoverflow.com/questions/30452395/selenium-pdf-automatic-download-not-working\
    pdflist = []
    
    if samp is None: 
        for a in urllist:
            getURL = driver.get(a)
            #time.sleep(2)
            #driver.implicitly_wait(2)
            link = WebDriverWait(driver, 100).until(EC.presence_of_element_located((By.XPATH, "/html/body/p[5]/a[1]"))) #xml path of the link taking us to the pdf 
            pdflist.append(link.get_attribute('href'))
            link.click()
      
        #download each pdf from the link identified in the "Live View" Session
        for a in pdflist:
            #inniate i 
            if i is None:
                i=0
            #pull back pdf from page, and save as pdf 
            urllib.request.urlretrieve(a,"file"+str(i)+"_"+max(a.split("/")[-1:10]))
            i = i+1     
    else: 
        for a in urllist[:samp]:
            getURL = driver.get(a)
            #time.sleep(2)
            #driver.implicitly_wait(2)
            link = WebDriverWait(driver, 100).until(EC.presence_of_element_located((By.XPATH, "/html/body/p[5]/a[1]"))) #xml path of the link taking us to the pdf 
            pdflist.append(link.get_attribute('href'))
            link.click()
            
        #download each pdf from the link identified in the "Live View" Session
        for a in pdflist[:samp]:
            if i is None:
                i=0

            #pull back pdf from page, and save as pdf 
            urllib.request.urlretrieve(a,"file"+str(i)+"_"+max(a.split("/")[-1:10]))
            i = i+1
          
    time.sleep(3)
    driver.quit()
    driver.quit()    
    
    
    
    
#add months to a date
def addmonths(date,months):
    targetmonth=months+date.month
    try:
        date.replace(year=date.year+int(targetmonth/12),month=(targetmonth%12))
    except:
        # There is an exception if the day of the month we're in does not exist in the target month
        # Go to the FIRST of the month AFTER, then go back one day.
        date.replace(year=date.year+int((targetmonth+1)/12),month=((targetmonth+1)%12),day=1)
        date+=datetime.timedelta(days=-1)

#rename a file from a specific folder
def tiny_file_rename(newname, folder_of_download):
    filename = max([f for f in os.listdir(folder_of_download)], key=lambda xa :   os.path.getctime(os.path.join(folder_of_download,xa)))
    if '.part' in filename:
        time.sleep(1)
        os.rename(os.path.join(folder_of_download, filename), os.path.join(folder_of_download, newname))
    else:
        os.rename(os.path.join(folder_of_download, filename),os.path.join(folder_of_download,newname))


#Find all files in a directory added in the last n minutes that are csv
#Find all files in a directory added in the last n minutes 
def recentfiles(path, lookback):
    files2 = []
    now = dt.datetime.now()
    ago = now-dt.timedelta(minutes=lookback)

    for root, dirs,files in os.walk(path):  
        for fname in files:
            path2 = os.path.join(root, fname)
            st = os.stat(path2)    
            mtime = dt.datetime.fromtimestamp(st.st_mtime)
            
            if os.path.splitext(path2)[1] == '.csv':        
                print('%s modified %s'%(path2, mtime))                
                files2.append(path2)
                break
    return(files2[0])  

#tfidf 
def tf_idf(site_list, search_terms):
    doc_list = []
    for site in site_list:
        doc_list.append([item for sublist in site for item in sublist])
    texts = [search_terms] + doc_list
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ## Using tf-idf conversion for corpus 
    tf_idf_model = models.TfidfModel(corpus,id2word=dictionary, normalize=True) # fit model
    tf_idf_corp = tf_idf_model[corpus] # apply model
    tf_idf_items = {"dic": dictionary, "tf_idf": tf_idf_corp}
    return tf_idf_items
	
	

#application/function to launch a spider from anywhere , even outside the locaiton of the project directory 
import os 
import subprocess 
from subprocess import PIPE, Popen 
def SpiderAnyWhere(scrapy_cmd=r"scrapy run FinraLinks -a entity_name='CitiGroup'"
,spider_dir=r"/home/docadmin/ZackC/NN_Gitlab/adhoc_troubleshoot/Finra/FinraLinks/spiders/FinraLinksCrawl.py"):
    #inputs
    here = os.path.abspath(os.path.dirname(spider_dir))
    cmd1 =scrapy_cmd
    process = subprocess.Popen([cmd1],shell=True,stdout=PIPE, cwd=here )
    output = process.stdout.read()
    result = output.decode('utf-8').strip().split('\n')
    return(result)
	
#ocrmypdf linux tool (best open source ocr) 
#ocrmypdf --output-type pdf --deskew --clean --rotate-pages --clean-final "+ UnSearchablePDFPaths[i]+" "+ pdf_output_path
##ocrmypdf --output-type pdf --deskew --clean --rotate-pages --clean-final "/home/docadmin/ZackC/Final OCR Scripts/OCR_Misc/test.pdf" "/home/docadmin/ZackC/Final OCR Scripts/OCR_Misc/out_test.pdf"
##pdftotext -layout "/home/docadmin/ZackC/Final OCR Scripts/OCR_Misc/out_test.pdf" "/home/docadmin/ZackC/Final OCR Scripts/OCR_Misc/out_test.txt"


#check if name is in sentences surrounding the sentence being processed (applied same way for entity and individual because all conditions are OR)  
def name_in_sent(s1,s2,s3,fname,lname):
	return True if fname in s1 or fname in s2 or fname in s3 or (lname!='' and (lname in s1 or lname in s2 or lname in s3)) else False

#extracts all sentences before and after the sentence being processed 
def previous_and_next(some_iterable):
	prevs, items, nexts = tee(some_iterable, 3)
	prevs = chain([''], prevs)
	nexts = chain(islice(nexts, 1, None), [''])
	return zip(prevs, items, nexts)

#FIND MOST SIMILAR SUBSTRING FROM A CORPUS 
#code summary: 
#    1. Scan the corpus for match values in steps of size step to find the approximate location of highest match value, pos.
#    2. Find the substring in the vicinity of pos with the highest match value, by adjusting the left/right positions of the substring.

##always keep step < len(query) * 3/4, and flex < len(query) / 3.
###SYNTAX#
##match = get_best_match("Zacks is bakery", "Zacks Bakery Has Great View of the city and never sleeps", step= 3,flex=10)
##print(match)
##match = get_best_match("CitiGroup Global Markets", "Zacks Bakery CitiGroup Has Great View of the city and never sleeps", step= 1,flex=10)
##print(match)
#import difflib
from difflib import SequenceMatcher
import sys

def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
    query : str
    corpus : str
    step : int
        Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    flex : int
        Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.

    Outputs
    -------
    output0 : str
        Best matching substring.
    output1 : float
        Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[int(p_l / step)]
        bmv_r = match_values[int(p_l / step)]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value



	
#Date extraction techniques 
import requests
import re
from bs4 import BeautifulSoup
def try_find_date(text,soup2):
    soup = soup2
#     obj=[]
    try:
        obj = soup.find(re.compile(r'meta'), property=re.compile(".*" + text+".*"))
    except:
        pass

    if not obj:
        try:
            obj = soup.find(re.compile(r'meta'), {"name":re.compile(".*" + text+".*")})
        except:
            pass
        
    if not obj:
        try:
            obj = soup.find(re.compile(r'meta'), {"itemprop":re.compile(".*" + text+".*")})
        except:
            pass
        
    if not obj:
        try:
            obj = soup.find(re.compile(r'meta'), {re.compile(".*" + text+".*"):re.compile(".*" + text+".*")})
        except:
            pass
        
    if not obj:
        try:
            obj = soup.find(re.compile(r'.*'), {re.compile(".*" + text+".*"):re.compile(".*" + text+".*")})
        except:
            pass

    if not obj:
        try:
            obj = soup.find(re.compile(r'.*time.*'))
        except:
            pass
        
    if not obj:
        try:
            obj = soup.find(re.compile(r'.*date.*'))
        except:
            pass
        
    return obj


def date_searchin(url1):
#'''SYNTAX:date_searchin('https://harvardlawreview.org/2015/02/sec-v-citigroup-global-markets-inc/')'''

    response1 = requests.get(url1)
    soup2 = BeautifulSoup(response1.content,"html.parser")
    search_fields = ['date','time','Date','Time']
    out_obj = []
    obj=[]
    for term in search_fields:
        if not obj: 
            obj = try_find_date(term,soup2)
        try: 
            out_obj = obj["content"]
        except:
            pass
        if not out_obj:
            try:
                for objj in obj.attrs:
                    if 'time' in objj:
                        out_obj.append(obj[objj])
            except:
                pass
        if not out_obj:
            try:
                for objj in obj.attrs:
                    if 'date' in objj:
                        out_obj.append(obj[objj])
            except:
                pass            
    print(out_obj)
    return(out_obj)
    
    #add a call out for not validate etc
    
    

##########TFIDF FUNCIONS############
#building tfidf with pipelines: https://buhrmann.github.io/sklearn-pipelines.html
#evaluating tfidf: https://buhrmann.github.io/tfidf-analysis.html

#extract top n terms from tfidf fitted object 
def topn_tfidf_freq(tfidfvectorizer,tfidf_fit_transform,n=20):
    #''' 
    #Source:http://www.ultravioletanalytics.com/blog/tf-idf-basics-with-pandas-scikit-learn
    #extract top n terms from tfidf matrix using the vectorizer object an the tfidf maxtrix fit on the data(tfidf_fit_transform)
    #inputs
    #tfidfvectorizer: object f type TfidfVectorizer() fom sklearn 
    #tfidf_fit_transform: obect of type TfidfVectorizer.fit.transform()
    #output: freq table oftop n items in tfidf ad weights associated with top n terms 
    #'''
    occ = np.asarray(tfidf_fit_transform.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': tfidfvectorizer.get_feature_names(), 'occurrences': occ})
    freqtable=counts_df.sort_values(by='occurrences', ascending=False).head(n)
    
    weights = np.asarray(tfidf_fit_transform.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tfidfvectorizer.get_feature_names(), 'weight': weights})
    weights_dfout=weights_df.sort_values(by='weight', ascending=False).head(n)

    return freqtable, weights_dfout

#function that takes a single row of the tf-idf matrix (corresponding to a particular document)
#and return the n highest scoring words (or more generally tokens or features):
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

#in order to apply the above function to inspect a particular document, we convert a single row into dense format first:
def top_feats_in_doc(Xtr, features, row_id, top_n=25):
#''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

#calculate the average tf-idf score of all words across a number of documents (in this case all documents)
# i.e. the average per column of a tf-idf matrix
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

#calculate the mean tf-idf scores depending on a document’s class label
def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

#plot tfidf 
def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    
#VISUALIZE TFIDF MATRIX 
def tfidf_visuals(tf_idf_matrix
                  , num_clusters=10
                  , num_seeds=10
                  , max_iterations=300
                  , pca_num_components = 2
                  , tsne_num_components=2
                  , labels_color_map = {0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'}                                       
                                       ):
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # calculate tf-idf of texts
    #tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
    #tf_idf_matrix = tf_idf_vectorizer.fit_transform(texts_list)
    
    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        precompute_distances="auto",
        n_jobs=-1
    )
    
    labels = clustering_model.fit_predict(tf_idf_matrix)
    # print labels
    
    X = tf_idf_matrix.todense()
    
    # ----------------------------------------------------------------------------------------------------------------------
    
    reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
    # print reduced_data
    
    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.show()
    
    if tsne ==True:
        # t-SNE plot
        embeddings = TSNE(n_components=tsne_num_components)
        Y = embeddings.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.show()

    return 'plot complete'


    
#model architecture for shallow neural network used in DOcumentClassificationTechniques.py
from keras import layers, models, optimizers

def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)
    
    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

#model architecture for CNN used in DocumentClassifcationTechniques.py
def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

#model architecture for LSTM in DocutmentCLassificationTechniques.py
def create_rnn_lstm(): 
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

def create_rnn_gru(): 
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model
