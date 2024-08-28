import logging


def get_logger(path, *args, **kwargs):
    logging.basicConfig(format = '%(asctime)s %(message)s',
                    datefmt = '%m-%d-%Y %H:%M:%S',
                    handlers=[
                        logging.FileHandler(path),
                        logging.StreamHandler()
                    ],
                    level=logging.DEBUG)
    return logging
