from dotenv import load_dotenv
import os
import logging


def read_env_variable(var_name):
    load_dotenv()
    value = os.getenv(var_name)
    if value is None:
        logging.error(f"Environment variable {var_name} is not set")
        raise ValueError(f"Please set the environment variable {var_name}")
    return value
