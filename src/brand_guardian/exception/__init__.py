import sys


def _error_message_detail(error: Exception, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] at line [{line_number}]: {error}"


class BrandGuardianException(Exception):
    """Wraps an underlying exception with file/line context for easier debugging."""

    def __init__(self, error: Exception, error_detail=sys):
        super().__init__(_error_message_detail(error, error_detail))
        self.original_error = error
