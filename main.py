from Speaking_Silence import logger
from Speaking_Silence.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Speaking_Silence.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Speaking_Silence.pipeline.stage_03_model_training import ModelTrainingPipeline

STAGE_NAME = "data_ingestion"

try:
    logger.info(f"----"*47)
    logger.info(f">>>> {STAGE_NAME} Starting <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> {STAGE_NAME} Success <<<<")
    logger.info(f"----"*47)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "prepare_base_model"

try:
    logger.info(f"----"*47)
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logger.info(f"----"*47)
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Train model"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e