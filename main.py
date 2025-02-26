import argparse
import mlflow
import mlflow.sklearn
from src.model_pipeline import (
    prepare_data,  # Use the prepare_data function directly
    train_decision_tree, evaluate_model, save_model, load_model, get_hyperparameters
)

def main():
    # Set the MLflow tracking URI to a local MLflow server
    mlflow.set_tracking_uri('http://localhost:5000')

    # Create or set an experiment
    experiment_name = "DecisionTreeExperiment"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_name)

    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser()
    
    # Add CLI arguments
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the dataset")

    # Parse the arguments
    args = parser.parse_args()

    # If --prepare is specified, prepare the data
    if args.prepare:
        X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(args.file_path)
        print("Data prepared.")

    # If --train is specified, train the model
    if args.train:
        # Prepare the data
        X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(args.file_path)

        # Start an MLflow run
        with mlflow.start_run():
            # Train the model using train_decision_tree
            model = train_decision_tree(X_train, y_train)
            
            # Log the model in MLflow
            mlflow.sklearn.log_model(model, "decision_tree_model")

            # Log parameters using get_hyperparameters
            hyperparameters = get_hyperparameters()
            mlflow.log_param('model_type', 'decision_tree')
            for param, value in hyperparameters.items():
                mlflow.log_param(param, value)

            # Evaluate the model and log accuracy
            accuracy, *_ = evaluate_model(model, X_test, y_test)  # Extract only accuracy
            mlflow.log_metric('accuracy', accuracy)

            # Save the model locally
            save_model(model, 'decision_tree_model.joblib')

            print(f"Model trained and saved. Accuracy: {accuracy}")

    # If --evaluate is specified, evaluate the model
    if args.evaluate:
        # Prepare the data
        X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(args.file_path)

        # Load the trained model
        model = load_model('decision_tree_model.joblib')

        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        print("Model evaluated.")

# Execute the main function if this file is run as a script
if __name__ == "__main__":
    main()