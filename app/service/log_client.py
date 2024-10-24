import logging
import os

class FolderNameFilter(logging.Filter):
    def filter(self, record):
        folder_name = os.path.basename(os.path.dirname(record.pathname))
        record.folder_and_filename = f"{folder_name}/{record.filename}"
        return True

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(folder_and_filename)s:%(lineno)d - %(message)s',
    )
    
    logger = logging.getLogger()
    
    folder_filter = FolderNameFilter()
    logger.addFilter(folder_filter)
    
    for handler in logger.handlers:
        handler.addFilter(folder_filter)
    
    return logger

# Create a global logger variable
logger = setup_logger()
