from Speaking_Silence.config.configuration import ConfigurationManager
from Speaking_Silence.components.model_evaluation import Evaluation
from Speaking_Silence import logger


STAGE_NAME = "Evaluation model"

class ModelEvaluationPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} Starting <<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>> {STAGE_NAME} Success <<<<")
    except Exception as e:
        logger.exception(e)
        raise e
