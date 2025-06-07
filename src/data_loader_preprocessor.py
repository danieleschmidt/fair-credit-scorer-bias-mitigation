import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_credit_data(file_path, protected_attribute_col, target_col):
    """
    Loads credit data, preprocesses it, and separates features, target, and protected attribute.

    Args:
        file_path (str): Path to the CSV file.
        protected_attribute_col (str): Name of the protected attribute column.
        target_col (str): Name of the target variable column.

    Returns:
        tuple: (X, y, protected_attribute_series)
               X (pd.DataFrame): Processed feature matrix.
               y (pd.Series): Target variable.
               protected_attribute_series (pd.Series): Protected attribute.
               Returns (None, None, None) if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataframe.")
        return None, None, None
    if protected_attribute_col not in df.columns:
        print(f"Error: Protected attribute column '{protected_attribute_col}' not found in the dataframe.")
        return None, None, None

    y = df[target_col]
    protected_attribute_series = df[protected_attribute_col]

    # Drop target and protected attribute from main dataframe for X processing
    # Make a copy to avoid SettingWithCopyWarning
    X_df = df.drop(columns=[target_col, protected_attribute_col]).copy()

    # Identify numerical and categorical columns
    numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure no overlap (though select_dtypes should handle this)
    numerical_cols = [col for col in numerical_cols if col not in [target_col, protected_attribute_col]]
    categorical_cols = [col for col in categorical_cols if col not in [target_col, protected_attribute_col]]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first') # drop='first' to avoid multicollinearity

    # Create a column transformer to apply transformations
    # Only include columns that are actually present in X_df
    transformers = []
    if numerical_cols:
        transformers.append(('num', numerical_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    if not transformers:
        print("No features to process after removing target and protected attribute.")
        # If X_df is empty or only contains columns that were removed
        return X_df, y, protected_attribute_series


    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    try:
        X_processed = preprocessor.fit_transform(X_df)

        # Rely ONLY on get_feature_names_out, assuming modern scikit-learn
        # This method is generally available and robust.
        processed_feature_names = preprocessor.get_feature_names_out()

        # Convert X_processed (NumPy array or sparse matrix) to DataFrame
        if hasattr(X_processed, "toarray"): # It's a sparse matrix
            data_for_df = X_processed.toarray()
        else: # It's already a dense numpy array
            data_for_df = X_processed

        X_processed_df = pd.DataFrame(data=data_for_df, columns=processed_feature_names)

        # Ensure column names are unique if any duplication occurred (e.g. from passthrough or complex feature names)
        if X_processed_df.columns.has_duplicates:
            cols = X_processed_df.columns.tolist()
            new_cols = []
            counts = {} # To keep track of occurrences of each column name
            for col_name in cols:
                if col_name in counts: # Check if name already processed
                    counts[col_name] += 1
                    new_cols.append(f"{col_name}_{counts[col_name]}")
                else:
                    # Check if this name (even without suffix) already exists due to other processing steps
                    # This also handles the first time we see a name that might eventually get a suffix
                    if cols.count(col_name) > 1:
                        counts[col_name] = 0
                        new_cols.append(col_name) # Append original name for the first time
                    else:
                        # If name is already unique, no need to track for suffixing
                        new_cols.append(col_name)
            X_processed_df.columns = new_cols

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # Add more debug information
        print(f"Type of X_df: {type(X_df)}")
        if isinstance(X_df, pd.DataFrame):
            print(f"X_df dtypes: \n{X_df.dtypes}")
            print(f"X_df head: \n{X_df.head()}")
            print(f"X_df shape: {X_df.shape}")
        print(f"Numerical cols identified: {numerical_cols}")
        print(f"Categorical cols identified: {categorical_cols}")
        print(f"Type of X_processed: {type(X_processed)}")
        if hasattr(X_processed, "shape"):
            print(f"Shape of X_processed: {X_processed.shape}")
        if 'processed_feature_names' in locals():
             print(f"Generated feature names: {processed_feature_names}")
        return None, None, None

    return X_processed_df, y, protected_attribute_series

if __name__ == '__main__':
    # Example Usage (assuming credit_data.csv is in data directory relative to script execution)
    # Corrected path for running from /app directory
    file_path = 'data/credit_data.csv'
    # Create dummy data for testing if the file doesn't exist
    try:
        # Create data directory if it doesn't exist for the main example
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"'{file_path}' not found. Creating a dummy one for testing the script.")
        dummy_data = {
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'credit_risk': [0, 1, 0, 1, 0],
            'age': [25, 34, 45, 22, 58],
            'loan_amount': [5000, 10000, 15000, 2000, 8000],
            'employment_duration': [2, 5, 10, 1, 20],
            'education': ['Bachelor', 'Master', 'PhD', 'High School', 'Bachelor'],
            'income_source': ['Salary', 'Freelance', 'Salary', 'Salary', 'Pension'] # another categorical
        }
        dummy_df = pd.DataFrame(dummy_data)
        # data directory is already checked/created above
        dummy_df.to_csv(file_path, index=False)
        print(f"Dummy '{file_path}' created.")


    target_col = 'credit_risk'
    protected_attribute_col = 'gender'

    X, y, pa_series = load_and_preprocess_credit_data(file_path, protected_attribute_col, target_col)

    if X is not None:
        print("\nProcessed X head:")
        print(X.head())
        print("\ny head:")
        print(y.head())
        print("\nProtected Attribute Series head:")
        print(pa_series.head())
        print(f"\nX shape: {X.shape}")
        print(f"Protected attribute categories: {pa_series.unique()}")

    # Test with a different dataset structure or columns
    print("\n--- Testing with slightly different data ---")
    # Create a temporary different CSV for testing
    # Corrected path for temporary file
    temp_data_path = 'data/temp_credit_data_test.csv'
    temp_data = {
        'sex': ['M', 'F', 'M', 'F'],
        'approved': [1, 0, 1, 0],
        'client_age': [30, 40, 25, 50],
        'loan_val': [100, 200, 50, 300],
        'job_type': ['A', 'B', 'A', 'C']
    }
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv(temp_data_path, index=False)

    X_temp, y_temp, pa_temp = load_and_preprocess_credit_data(temp_data_path, 'sex', 'approved')
    if X_temp is not None:
        print("\nProcessed X_temp head:")
        print(X_temp.head())
        print(f"\nX_temp shape: {X_temp.shape}")
        print(f"Protected attribute (sex) categories: {pa_temp.unique()}")

    # Clean up temporary file
    import os
    try:
        os.remove(temp_data_path)
    except OSError as e:
        print(f"Error removing temporary file {temp_data_path}: {e}")

    # Test case: what if protected attribute is numerical?
    print("\n--- Testing with numerical protected attribute (should pass it through) ---")
    # Corrected path for temporary file
    temp_data_num_pa_path = 'data/temp_credit_data_num_pa.csv'
    temp_data_num_pa = {
        'group_id': [101, 102, 101, 103, 102],
        'outcome': [0, 1, 0, 1, 0],
        'feature1': [25, 34, 45, 22, 58],
        'feature2': [5000, 10000, 15000, 2000, 8000],
        'category_feature': ['X', 'Y', 'X', 'Y', 'Z']
    }
    temp_df_num_pa = pd.DataFrame(temp_data_num_pa)
    temp_df_num_pa.to_csv(temp_data_num_pa_path, index=False)

    X_num_pa, y_num_pa, pa_num_pa = load_and_preprocess_credit_data(temp_data_num_pa_path, 'group_id', 'outcome')
    if X_num_pa is not None:
        print("\nProcessed X_num_pa head:")
        print(X_num_pa.head())
        print(f"\nX_num_pa shape: {X_num_pa.shape}")
        print(f"Protected attribute (group_id) head: \n{pa_num_pa.head()}")
        print(f"Protected attribute (group_id) dtype: {pa_num_pa.dtype}")

    try:
        os.remove(temp_data_num_pa_path)
    except OSError as e:
        print(f"Error removing temporary file {temp_data_num_pa_path}: {e}")

    print("\n--- Testing with file not found ---")
    X_fnf, y_fnf, pa_fnf = load_and_preprocess_credit_data('data/non_existent_file.csv', 'gender', 'credit_risk')
    # Expected: Error message and None returns

    print("\n--- Testing with target column not found ---")
    X_tcf, y_tcf, pa_tcf = load_and_preprocess_credit_data(file_path, 'gender', 'non_existent_target')
    # Expected: Error message and None returns

    print("\n--- Testing with only numerical features (after dropping target/protected) ---")
    # Corrected path for temporary file
    temp_data_only_num_path = 'data/temp_credit_data_only_num.csv'
    temp_data_only_num = {
        'prot_attr': ['A', 'B', 'A', 'B'],
        'target_var': [0, 1, 0, 1],
        'num_feat1': [10, 20, 30, 40],
        'num_feat2': [0.5, 0.1, 0.8, 0.2]
    }
    temp_df_only_num = pd.DataFrame(temp_data_only_num)
    temp_df_only_num.to_csv(temp_data_only_num_path, index=False)
    X_on, y_on, pa_on = load_and_preprocess_credit_data(temp_data_only_num_path, 'prot_attr', 'target_var')
    if X_on is not None:
        print(X_on.head())
        print(X_on.shape)
    try:
        os.remove(temp_data_only_num_path)
    except OSError as e:
        print(f"Error removing temporary file {temp_data_only_num_path}: {e}")


    print("\n--- Testing with only categorical features (after dropping target/protected) ---")
    # Corrected path for temporary file
    temp_data_only_cat_path = 'data/temp_credit_data_only_cat.csv'
    temp_data_only_cat = {
        'prot_attr_cat': [1, 0, 1, 0], # Numerical protected attribute
        'target_var_cat': [0, 1, 0, 1],
        'cat_feat1': ['P', 'Q', 'R', 'P'],
        'cat_feat2': ['X', 'Y', 'X', 'Z']
    }
    temp_df_only_cat = pd.DataFrame(temp_data_only_cat)
    temp_df_only_cat.to_csv(temp_data_only_cat_path, index=False)
    X_oc, y_oc, pa_oc = load_and_preprocess_credit_data(temp_data_only_cat_path, 'prot_attr_cat', 'target_var_cat')
    if X_oc is not None:
        print(X_oc.head())
        print(X_oc.shape)
    try:
        os.remove(temp_data_only_cat_path)
    except OSError as e:
        print(f"Error removing temporary file {temp_data_only_cat_path}: {e}")
