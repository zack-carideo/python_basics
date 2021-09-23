import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import connection, Pool
import os, sys
from typing import List, Union, Dict
from functools import partial
import logging
import pickle

from text_processing.text_preprocessing import clean_text_str

#prints loggs from each worker to terminal for debugging or QA
multiprocessing.log_to_stderr(level=logging.DEBUG)

'''
helper functions to assist in the distribution and aggregation of data pushed to and from multi-processing workers 
'''

def create_batch(iterable
                 , split_info={'v1': 'scenario','v2':'date'}
                 , batch_size=10):

    """splits an input iterable into batches to be passed to workers based on the logic in split_info(optional) or based on batch_size if split info is not provided
    @param iterable(required): any iterable (pd.DataFrame, List[anything], Dict, ...)
    @param split_info(optional): dictionary of critera upon which to use in splitting the input iterable into batches (s1 = split1 , s2 = split2, but you must pass dictionary with keys s1,s2, or s1 and s2
    @param batch_size(required if split_info not given): batch size to randomly split input interable into for downstream mp workers
    @return: yields an generartor that is only evaluated when the function is called (aka its a iterable that is returned in form of generator to minimze memory overhead)

    example
    a = create_batch(df, batch_size=10) #splits data into chunks of 10 randomly
    a = create_batch(df, split_info = {'split1':'scenario','split2':'pd_bucket'}) #splits data into chunks based on scenario and pdbucket
    """


    # how big is the input iterable
    length = len(iterable)

    if split_info is not None:
        # assert the split_info dictionary has correct keys (if it is not None)
        assert all(key in ['v1','v2']  for key in split_info.keys()), 'only v1 and v2 are valid key names to use in ' \
                                                                      'split_info dict. they id the variables to be ' \
                                                                      'used in generating stratified partitions '

        if len(list(split_info.keys())) == 1:
            for val in iterable[split_info['v1']].unique():
                yield iterable[iterable[split_info['v1']] == val]

        elif len(split_info.keys()) == 2:
            for val1 in iterable[split_info['v1']].unique():
                for val2 in iterable[split_info['v2']].unique():
                    yield iterable[((iterable[split_info['v1']] == val1) & (iterable[split_info['v2']] == val2))]
        else:
            raise NotImplementedError(
                'input iterable splitting can only be stratified on 1 or 2 variables, 3+ variable stratification is '
                'not implemented ')
    else:
        for ndx in range(0,length,batch_size):
            yield iterable[ndx:(ndx+batch_size)]



def bytes2df(bytes_object):
    """convert bytes representation of parquet file , that has been saved to the buffer for quick retreval (when you compress a parquet into bytes
     this is how you re-construct the original data, very helpful when there are limts size of data that
     a 'process' can communicate to another with (the memory overhead)

    @param bytes_object:a parquet or any file type converted into bytes format
    @return: a pandas dataframe representation of the bytes object ( a normal table/dataframe)
    """
    reader = pa.BufferReader(bytes_object)
    table = pq.read_table(reader)
    return table.to_pandas()


'''
using async_pool to parallelize function with MANY inputs
'''


# sample functions to use in mp as callabacks
# primary function called in pool_aysnc()
def read_me(in_path, abc):
    df = pd.read_parquet(in_path)
    df['tst'] = abc
    return df


# callback function used in pool_async()
def save_me(df, out_path, bytes_in=False):
    if bytes_in:
        out_df = bytes2df(df)
        out_df.to_parquet(out_path)
    else:
        df.to_parquet(out_path)

def write_to_hdfs(path_info: List[Dict], async_func, callback):
    '''

    @param path_info: [{'in_path': <file 2 load> , 'out_path':<path 2 save loaded file>}]
    @param async_func: function to execute in async fastion (primary function passed to worker)
    @param callback: callback to execute post return from inital async job
    @return:
    '''

    # create a pool of processes, 1 for each item in the path_info_list
    # WARNING: NO LIMIT ON TOTAL PROCESSES , THIS COULD EAT ALL CPU RESOURCES
    pool = Pool(processes=len(path_info), maxtasksperchild=1)

    # LOAD DATA -> PASS IN A PATH -> RETURN A DATAFRAME -> SAVE THE DATAFRAME IN A CALLBACK
    for p_info in path_info:

        # save the output to a file using callback (PARTIAL USED TO ADD A  PARAMETER TO THE CALLBACK FUNCTION THAT IS
        # EXECUTED ON THE OUTPUT OF THE ORIGINAL ASYNC FUNCTION)
        new_callback = partial(callback, out_path=p_info['out_path'])

        # read test file to save
        pool.apply_async(
            async_func
            , args=[p_info['in_path'], 'abcd']  # testing if we can pass multiple positional parameters to pool_async
            , callback=new_callback
        )
    pool.close()
    pool.join()


'''
using mp.process to parallelize a single function and return all results to 1 object
'''

# mp.Process version of clean text
def clean_text_mp(text_list, str_i, return_dict):
    '''function to clean, and save results to a mp.dictionary for use in aggregating worker outpouts when mp.process is used

    @param text_list: list of text to parse ['abc21','adfa adfi aa df',...]
    @param str_i: str_i used to track which process executed which text lists
    @param return_dict: mp dictionayr used to save store aggregate and return outputs executed by mp.Process
    @return: mp.dict
    '''

    #create container to hold output
    out_list = [clean_text_str(text) for text in text_list]

    #return statement required to pass mp dictionary to and from processes to aggregate / populate output
    return_dict[str_i] = [val for val in out_list ]


def clean_text_mp_run(text_lists,out_path = None, save_output=False):
    '''

    @param text_lists: list of lists of texts , each list representation a partition of items to be passed to a
    spawned process
    @return: @NOTE: the order in which the process's receive the data is NOT the same order the data will
    be saved in the return_dict(). use str_i to ensure input order matches output order
    '''

    #Set up multiprocessing manager and dictionary to pass to each process spawned
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    i=0

    for _list in text_lists:
        p = multiprocessing.Process(target = clean_text_mp, args = (_list, str(i),return_dict))
        jobs.append(p)
        p.start()
        i+=1

    for proc in jobs:
        proc.join()

    #aggregate output back into single object
    clean_text = []
    for key in return_dict.keys():
        clean_text = clean_text + return_dict[key]

    if save_output:
        assert out_path is not None, 'out_path must be provided if you choose to save the output'
        with open(out_path,'wb') as outfit:
            pickle.dump(clean_text, outfit)

    else:
        return clean_text



if __name__ == '__main__':

    #test input list of strings to clean and save
    texts = ['i am zack','i am jack','i am matt','i am dan ','i ate ham ','i have a fan','your my man','run into the man','with the sand','for a band','in the van']
    out_path = 'C:/Users/zjc10/Desktop/Projects/data/example_data/outputs/test_mp_stringclean.pkl'

    #split input into list of list of chunks
    batches = create_batch(texts
                           , split_info=None
                           , batch_size=2 )

    #execute the function over the partitions of data using pool.Process()
    out_list = clean_text_mp_run(batches
                                 , out_path = out_path
                                 , save_output=True)

    #validate it worked
    with open(out_path, 'rb') as f:
        a = pickle.load(f)
    print(a)