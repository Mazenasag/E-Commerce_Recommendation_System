import sys
import traceback


def error_custom_details(error):
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        # Fallback if no traceback available
        tb_list = traceback.extract_tb(exc_tb) if exc_tb else []
        if tb_list:
            file_name = tb_list[-1].filename
            line_number = tb_list[-1].lineno
        else:
            file_name = "unknown"
            line_number = 0
    
    error_message = "Error occurred in python script name [{0}] in line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details=None):
        super().__init__(error_message)
        self.error_message = error_custom_details(error_message)

    def __str__(self):
        return self.error_message