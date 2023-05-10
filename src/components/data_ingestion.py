import os
import sys
from src.exception import CustomExection
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        logging.info("DataIngestion init method called")
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info('Read data as dataframe from csv file')

            logging.info(os.path.dirname(self.ingestion_config.train_data_path))

            os.mkdir(os.path.dirname(self.ingestion_config.train_data_path))
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Saving training dataset in to csv file")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Saving test dataset in to csv file")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Done with train and test data split')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.error('Error has occoured')
            logging.error(e)
            raise CustomExection(e, sys)
           

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

