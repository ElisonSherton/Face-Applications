import sys, logging
import pytz
from pytz import timezone
from datetime import datetime

def get_logger(self):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    def timetz(*args):
        return datetime.now(tz).timetuple()

    tz = timezone('Asia/Kolkata')
    logging.Formatter.converter = timetz

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    filename = datetime.now().astimezone(timezone("Asia/Kolkata")).strftime("logs/training_%d_%m_%Y_%a_%H_%M_%S.log")

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger