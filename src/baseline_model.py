import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_baseline_classifier(X_train, y_train, X_test, y_test):
    """
    Initializes, trains, and evaluates a Logistic Regression model.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target variable.
        X_test (pd.DataFrame or np.ndarray): Testing features.
        y_test (pd.Series or np.ndarray): Testing target variable.

    Returns:
        tuple: (model, accuracy)
               model (LogisticRegression): The trained Logistic Regression model.
               accuracy (float): The accuracy of the model on the test set.
    """
    # Initialize the Logistic Regression model
    # Using solver='liblinear' which is good for small datasets and binary classification.
    # random_state for reproducibility of results if solver involves randomness (e.g. 'sag', 'saga')
    # For liblinear, random_state is not strictly necessary but doesn't harm.
    model = LogisticRegression(solver='liblinear', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

if __name__ == '__main__':
    print("Running baseline model script example...")

    # Generate some dummy data for demonstration
    # X will have 100 samples, 5 features
    X_dummy = np.random.rand(100, 5)
    # y will be binary target variable (0 or 1)
    y_dummy = np.random.randint(0, 2, 100)

    # Split data into training and testing sets
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42
    )

    print(f"X_train_dummy shape: {X_train_dummy.shape}")
    print(f"y_train_dummy shape: {y_train_dummy.shape}")
    print(f"X_test_dummy shape: {X_test_dummy.shape}")
    print(f"y_test_dummy shape: {y_test_dummy.shape}")

    # Train and evaluate the baseline classifier
    trained_model, model_accuracy = train_baseline_classifier(
        X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy
    )

    print(f"\nTrained Logistic Regression Model: {trained_model}")
    print(f"Model Accuracy on Dummy Data: {model_accuracy:.4f}")

    # Example with slightly different data (e.g., more features)
    X_dummy_more_features = np.random.rand(100, 10)
    y_dummy_more_features = np.random.randint(0, 2, 100)
    X_train_mf, X_test_mf, y_train_mf, y_test_mf = train_test_split(
        X_dummy_more_features, y_dummy_more_features, test_size=0.3, random_state=123
    )
    
    model_mf, acc_mf = train_baseline_classifier(X_train_mf, y_train_mf, X_test_mf, y_test_mf)
    print(f"\nTrained Model (10 features): {model_mf}")
    print(f"Accuracy (10 features): {acc_mf:.4f}")

    # Test with minimal data (e.g. very few samples)
    # This can sometimes cause issues with some solvers or if data is not diverse enough
    # LogisticRegression with liblinear should be fairly robust.
    X_tiny = np.array([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]])
    y_tiny = np.array([0,1,0,1,0,1])
    if len(np.unique(y_tiny)) > 1 : # Ensure there's more than one class for splitting
        X_train_tiny, X_test_tiny, y_train_tiny, y_test_tiny = train_test_split(
             X_tiny, y_tiny, test_size=0.33, random_state=42 # 2 samples for test
        )
        if X_train_tiny.shape[0] > 0 and X_test_tiny.shape[0] > 0:
             model_tiny, acc_tiny = train_baseline_classifier(X_train_tiny, y_train_tiny, X_test_tiny, y_test_tiny)
             print(f"\nTrained Model (tiny data): {model_tiny}")
             print(f"Accuracy (tiny data): {acc_tiny:.4f}")
        else:
            print("\nNot enough data to form train/test for tiny dataset after split.")
    else:
        print("\nTiny dataset has only one class, cannot train/test meaningfully.")
