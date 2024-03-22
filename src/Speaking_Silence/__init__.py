import os
import sys
import logging
import colorlog

# Define the logger string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create the logs directory
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Define colorful formatter
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s: %(levelname)s: %(module)s]%(reset)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    reset=True,  # Reset color after log_color
)

# Configure logging
logger = logging.getLogger("Speaking_Silence")
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_filepath)
file_handler.setFormatter(logging.Formatter(logging_str))
logger.addHandler(file_handler)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

