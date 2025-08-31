import yaml
from loguru import logger


def replace_empty_string(row):
    if row == "":
        return {"score": 0, "evidences": []}
    else:
        return row


def load_yaml_file(file_path: str):
    try:
        with open(file_path, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        logger.error(f"Error loading yaml file: {e}")
        return None