from Speaking_Silence import logger
from  Speaking_Silence.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "data_ingestion"

try:
    logger.info(f">>>> {STAGE_NAME} Starting <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> {STAGE_NAME} Success <<<<")
except Exception as e:
    logger.exception(e)
    raise e
