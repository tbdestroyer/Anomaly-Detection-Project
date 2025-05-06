from clearml import Task

def init_clearml(project_name, task_name):
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = task.get_logger()
    return task, logger

def log_classification_metrics(logger, report_dict, accuracy, auc_score, threshold, cm, model_name):
    logger.report_scalar("Overall", "Accuracy", iteration=0, value=accuracy)
    logger.report_scalar("Overall", "ROC AUC", iteration=0, value=auc_score)
    logger.report_scalar("Overall", "Threshold", iteration=0, value=threshold)

    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                logger.report_scalar(f"{label}", metric_name, iteration=0, value=value)

    logger.report_confusion_matrix(
        title=f"{model_name} Confusion Matrix",
        series=model_name,
        matrix=cm.tolist(),
        iteration=0
    )
