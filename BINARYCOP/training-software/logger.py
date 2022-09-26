import logging
#set different formats for logging output
console_logging_format = '%(levelname)s: %(message)s'
file_logging_format = '%(levelname)s: %(asctime)s: %(message)s'

# configure logger
logging.basicConfig(level=logging.DEBUG, format=console_logging_format)
logger = logging.getLogger()
# create a file handler for output file
handler = logging.FileHandler('console_and_file.log')
 
# set the logging level for log file
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter(file_logging_format)
handler.setFormatter(formatter)
 
# add the handlers to the logger
logger.addHandler(handler)
 
# output logging messages
logger.info("Logger is initialised")