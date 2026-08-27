import sys
def error_message_details(error,error_details:sys):
    # Get exception details (type, value, traceback)
    _,_,exc_tb=error_details.exc_info()
    # exc_tb = traceback object → contains info about where error happened



    # Get the filename where the exception occurred

    file_name=exc_tb.tb_frame.f_code.co_filename

    # Explanation:
    # exc_tb.tb_frame        → current stack frame where error happened
    # .f_code               → code object
    # .co_filename          → gives the file name of that code

    error_message=(
        "Error occured in pyhton file name [{0}]"
        "line number [{1}] error message [{2}]"

    ).format(
        file_name,
        exc_tb.tb_lineno,
        str(error)

    )


    return error_message

class CustomeException(Exception):  # Here we inherits from python's built-in exception
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(
            error_message,
            error_details 

        )
    
    def __str__(self):
        return self.error_message