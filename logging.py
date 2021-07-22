'''
BASIC CUSTOM LOGGING CLASS
- this enables you to control logging operations within and across custom class objects
- you can pass this logger across functions / classes and have them all conform to the same standard for logging
- logging to file can also be accommodated across processes, all can log to the same file
#'''

import logging
import os
from typing import  List, Dict
import multiprocessing as mp
import sys

# log_info = [{'path':<dir to save log>
#      , 'filename': <name of logfile>
#     , 'level': <log1 level>
#     , 'format': <log1_format>
#     , 'filter': <specific log1 level to filter on>}
#     , {'path':<path to save log2>
#         , 'level': <log2 level>
#          , 'format': <log3_format>
#          , 'filter': <specific log2 level to filter on>
#      }]

# EX SYNTAX
# log_info = [{'path':os.environ['Log_Dir']
#      , 'filename': 'test_log.log'
#     , 'level': 'INFO'
#     , 'format': '%(asctime)s %(levelname)-8s %(message)s'
#     , 'filter': 'DEBUG'
# }]

#stream_props = {'level': 'INFO', 'format': '%(asctime)s %(levelname)-8s %(message)s'}


#class to filter logging messages from specific level
class LogFilter(object):
    def __init__(self,level):
        self._level = level

    def filter(self, logRecord):
        return logRecord.levelno == self._level


class CustomLogger():

    """
    INPUTS
    # EX SYNTAX
     log_info = [{'path':os.environ['Log_Dir']
                  , 'filename': 'test_log.log'
                 , 'level': 'INFO'
                 , 'format': '%(asctime)s %(levelname)-8s %(message)s'
                 , 'filter': 'DEBUG'
                }]

    #stream_props = {'level': 'INFO', 'format': '%(asctime)s %(levelname)-8s %(message)s'}

    INITALIZATION
    #initalize custom logger
    logger = CustomLogger(log_info, stream=True, stream_props = stream_props, parallel=False)

    OUTPUT
    None (log written to file(s)
    """
    def __init__(self, log_info: List,stream:bool=True,  stream_props: Dict = None , parallel = False, *args, **kwargs ) -> None:

        #create standard logging object
        self.logger = logging.getLogger(__name__)

        #set level
        self.logger.setLevel(logging.DEBUG)


        for logfile in log_info:

            #get location to save log from input dict
            path = logfile.get('path',None)
            if path is None:
                raise NotADirectoryError('Please enter path to save log file to')

            #extract logfile information from dict, or apply default value if not provided
            filename = logfile.get('filename', f'{__name__}_run.log')
            level = logfile.get('level','INFO').upper()
            logformat = logfile.get('format', '%(asctime)s %(levelname)-8s %(message)s')
            filter = logfile.get('filter',None)

            #set up file handler
            fh = logging.Filehandler(os.path.join(path,filename),mode='w')
            fh.setLevel(logging.getLevelName(level))
            fh.setFormatter(logging.Formatter(logformat))

            if filter is not None:
                fh.addFilter(LogFilter(logging.getLevelName(filter)))

            #add the handler to the customer logging object
            self.logger.addHandler(fh)

            #set up logging to standard output if requested
            if stream:
                if stream_props is None:
                    raise ValueError('stream needs stream_props argument, a dictionary with keys "level" and "format" ')

                sh = logging.StreamHandler()
                level = stream_props.get('level','INFO').upper()
                sh.setLevel(logging.getLevelName(level))
                logformat = stream_props.get('format','%(asctime)s %(levelname)-8s %(message)s')
                sh.setFormatter(logging.Formatter(logformat))

                #add stream handler to custom logger
                self.logger.addHandler(sh)


            #set up multiprocessing que to handle logging across processes
            if parallel:
                self.parallel = True
                self.logQ = mp.Queue()
                self.listening = mp.Process(target = self.listener, args = (self.logQ))
                self.listening.start()
            else:
                self.parallel = False


    #mp que listener for handling logs across processes
    def listener(self, q: mp.Queue)-> None:

        while True:
            try:
                record = q.get()
                if record is None:
                    #use 'None' to end logging
                    logging.shutdown()
                    break
                self.logger.log(record[0], "%s", str(record[1]))

            except Exception:

                #check for errors in logger
                import sys
                import traceback
                print('problem with log: ', file= sys.stderr)
                traceback.print_exec(file=sys.stderr)

        return None

    def debug(self, message: str)->None:
        if self.parallel:
            self.logQ.put((logging.DEBUG, message))
        else:
            self.logger.debug(message)
        return None

    def info(self, message:str)-> None:
        if self.parallel:
            self.logQ.put((logging.INFO, message))
        else:
            self.logger.info(message)

        return None

    def warning(self, message: str)-> None:
        if self.parallel:
            self.logQ.put((logging.WARNING, message))
        else:
            self.logger.warning(message)

    def error(self, message: str)-> None:
        if self.parallel:
            self.logQ.put((logging.ERROR, message))
        else:
            self.logger.error(message)

    def critical(self, message: str) -> None:
        if self.parallel:
            self.logQ.put((logging.CRITICAL, message))
        else:
            self.logger.critical(message)

    def shutdown(self)-> None:
        if self.parallel:
            self.logQ.put(None)
            self.listening.join()
        else:
            logging.shutdown()





