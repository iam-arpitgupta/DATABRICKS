import pandas as pd
import os 
import yaml 
from loguru import logger 
from pyspark.sql import SparkSession 


from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")


logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

#load the data 
spark = SparkSession.builder.getOrCreate()

filepath =-
# Save to catalog "data/marverl_character/csv"

df = pd.read_csv(filepath)

# Display basic info about the dataset
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Columns: {list(df.columns)}")
logger.info(f"Target column '{config.target}' distribution:")
logger.info(df[config.target].value_counts())


data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

logger.info(f"Data preprocessing completed.")


# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()