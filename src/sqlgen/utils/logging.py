import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{self.GREEN}{record.msg}{self.END}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.YELLOW}{record.msg}{self.END}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.END}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{self.CYAN}{record.msg}{self.END}"
        
        return super().format(record)

def setup_colored_logging():
    """Set up colored logging for model calls"""
    logger = logging.getLogger()
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    return logger

