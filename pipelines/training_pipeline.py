from zenml import pipeline

@pipeline(enable_cache=False)
def train_pipeline(data_path: str) -> dict:
    """
    Train model pipeline that ingests, cleans, trains, and evaluates the model.

    Args:
        data_path: the path to the data
    Returns:
        Dictionary containing the r2_score and rmse_score
    """
    # data = ingest_data(data_path)
    
    # X_train, X_test, y_train, y_test = clean_data(data)

    # model = train_model(X_train, y_train, ModelNameConfig())

    # r2_score, rmse_score = evaluate_model(model, X_test, y_test)

    # return {'r2_score': r2_score, 'rmse_score': rmse_score}
    pass