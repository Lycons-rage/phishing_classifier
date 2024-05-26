import mlflow
from datetime import datetime

class Mlflow_logs:
    def __init__(self):
        self.current_timestamp = datetime.now()
        self.formatted_timestamp = self.current_timestamp.strftime("%Y%m%d_%H%M%S")
        self.base_name = "logs"
        self.extension = ".txt"

    def log_msg(self, message):
        filename = f"{self.base_name}_{self.formatted_timestamp}{self.extension}"
        mlflow.log_text(message, filename)