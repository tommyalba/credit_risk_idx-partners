import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#===================================

# Import Data while Handling Duplicate
def importData(data_path):
    # Read file
    dataframe = pd.read_csv(data_path)
    print(f"Origin Data Shape: {dataframe.shape} - (# Observation, # Column)")

    # Calculate the sum of duplicate data
    data_duplicates = dataframe.duplicated().sum()
    print(f"\n...handling duplicate data:")
    print(f"Sum of duplicate data: {data_duplicates}")
    
    # Get duplicated rows
    duplicated_rows = dataframe[dataframe.duplicated(keep=False)]
    print(f"Shape of duplicated rows: {duplicated_rows.shape}")
    
    # Print shape of DataFrame before dropping duplicates
    print(f"\nBefore drop rows: {dataframe.shape}")
    
    # Drop duplicates, keeping the first occurrence
    dataframe = dataframe.drop_duplicates(keep='first')
    
    # Print shape of DataFrame after dropping duplicates
    print(f"After drop rows: {dataframe.shape}")

    return dataframe


# Function to serialize data
def serialize_data(data, path: str) -> None:
    """
    Serialize data into a file using joblib.

    Parameters:
    data: The instance to be serialized (can be a DataFrame, Series, or other object).
    path (str): The file address where the data will be saved in pickle format.

    Returns:
    None: This function does not return a value, it only saves the data to a file.
    """
    joblib.dump(data, path)
    print(f"Data berhasil diserialisasi ke {path}")
    

# Function to deserialize data
def deserialize_data(path: str):
    """
    Deserialize data from a file using joblib.

    Parameters:
    path (str): The file address where the data is stored in pickle format.

    Returns:
    object: Deserialized data (can be a DataFrame, Series, or other object).
    """
    data = joblib.load(path)
    print(f"Data berhasil dideserialisasi dari {path}")
    return data

    
# Summary of Missing Values
def missing_summary(dataframe):
    """
    Creates a summary of missing values ​​for each column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be examined.
    
    Returns:
        pd.DataFrame: Summary of the number and percentage of missing values.
    """
    na_counts = dataframe.isna().sum()[dataframe.isna().sum() > 0]
    na_percentage = dataframe.isna().sum()[dataframe.isna().sum() > 0] / len(dataframe) * 100
    
    summary = pd.DataFrame({
        'Missing Values': na_counts,
        'Percentage': na_percentage.round(2),
        'Total length': len(dataframe)
    })
    
    return summary

# Function to Plot Numeric Distribution
def plot_numeric_distributions(
    dataframe, 
    target_col='y', save_plot=False,
    filename="numeric_distributions.png"):
    """
    Creates a distribution visualization (KDE + Boxplot) 
    for all numeric columns in a DataFrame against a target.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame input.
    target_col : str, default='y'
        The name of the target column to be used as the hue/axis.
    save_plot : bool, default=False
        If True, the graph will be saved to local storage.
    filename : str, default='numeric_distributions.png'
        The name of the saved image file (can include the folder path).
    """

    # Validasi target_col
    if isinstance(target_col, str):
        if target_col not in dataframe.columns:
            raise KeyError(f"Column '{target_col}' not found in DataFrame.")
        df_temp = dataframe.dropna()
        hue_col = target_col
    elif isinstance(target_col, (pd.Series, np.ndarray)):
        y_series = pd.Series(target_col, name='y')
        df_temp = dataframe.copy()
        df_temp['y'] = y_series
        df_temp = df_temp.dropna()
        hue_col = 'y'
    else:
        raise TypeError("`target_col` must be a String (column name) or Pandas Series / NumPy Array.")
    
    # Ambil semua kolom numerik (target sudah di-handle)
    num_cols = df_temp.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != hue_col]

    if len(num_cols) == 0:
        raise ValueError("There are no numeric columns to visualize.")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(len(num_cols), 2, figsize=(16, 6 * len(num_cols)))
    axes = np.atleast_2d(axes)  # pastikan selalu 2D

    for i, col in enumerate(num_cols):
        # KDE Plot
        sns.kdeplot(
            data=df_temp,
            x=col,
            hue=hue_col,
            fill=True,
            common_norm=False,
            palette={0: 'blue', 1: 'red'},
            alpha=0.5,
            ax=axes[i, 0]
        )
        axes[i, 0].set_title(f'Density Distribution of `{col}` by Target ({hue_col})', fontsize=13)

        # Boxplot
        sns.boxplot(
            data=df_temp,
            x=hue_col,
            y=col,
            palette={'0': 'blue', '1': 'red'},
            ax=axes[i, 1]
        )
        axes[i, 1].set_title(f'Median Comparison of `{col}` by Target ({hue_col}))', fontsize=13)

    plt.tight_layout()

    # Additional Logic: Save the image if the save_plot parameter is True
    if save_plot:
        # bbox_inches='tight' ensures that the label/text on the edge is not cut off when saved
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Visualization graph successfully saved as: {filename}")
        
    plt.show()

    
# WoE (Weight of Evidence) Function for Feature Selection
def calculate_all_iv(dataframe, target_col, num_bins=6):
    iv_results = {}

    # Input Scenario Validation `target_col`
    if isinstance(target_col, str):
        # Scenario A: `target_col` is the column name (string) in the dataframe.
        if target_col not in dataframe.columns:
            raise KeyError(f"Column '{target_col}' not found in DataFrame.")
        
        y_series = dataframe[target_col]
        X_dataframe = dataframe.drop(columns=[target_col])
    elif isinstance(target_col, (pd.Series, np.ndarray)):
        # Scenario B: `target_col` is a standalone Series or Array
        y_series = pd.Series(target_col) if isinstance(target_col, np.ndarray) else target_col
        X_dataframe = dataframe.copy()
    else:
        raise TypeError("`target_col` must be a String (column name) or Pandas Series / NumPy Array.")
        
    # Fetch all numeric columns except the target column itself.
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    total_good = (y_series == 0).sum()
    total_bad = (y_series == 1).sum()
    
    for col in numeric_cols:
        try:
            # Provide a copy of the data for internal calculations per column/feature
            temp_df = pd.DataFrame({
                'feature': X_dataframe[col],
                'target': y_series.values if hasattr(y_series, 'values') else y_series
            }).dropna() # delete temporary blank lines for IV count, if any
            
            # Automatic binning based on quantiles
            temp_df['bin'] = pd.qcut(temp_df['feature'], q=num_bins, duplicates='drop').astype(str)
            
            # Calculate aggregate
            aggr = temp_df.groupby('bin')['target'].agg(
                bad_counts='sum',
                total_counts='count'
            ).reset_index()
            
            aggr['good_counts'] = aggr['total_counts'] - aggr['bad_counts']
            
            # Distribution %
            aggr['dist_good'] = aggr['good_counts'] / total_good
            aggr['dist_bad'] = aggr['bad_counts'] / total_bad
            
            # Calculate WoE & IV
            aggr['WoE'] = np.log((aggr['dist_good'] + 1e-4) / (aggr['dist_bad'] + 1e-4))
            aggr['IV'] = (aggr['dist_good'] - aggr['dist_bad']) * aggr['WoE']
            
            # Total IV for this feature
            iv_results[col] = aggr['IV'].sum()
            
        except Exception as e:
            # Skip if column has too little value variation to qcut
            continue
            
    # Convert the results to a DataFrame and sort by highest IV
    iv_df = pd.DataFrame(list(iv_results.items()), columns=['Feature', 'Information Value (IV)'])
    return iv_df.sort_values(by='Information Value (IV)', ascending=False).reset_index(drop=True)


# Variance Inflation Factor
def compute_vif(dataframe, target_col=None):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    num_cols = dataframe.select_dtypes(include=['int64', 'float64', 'float32']).columns

    if target_col in num_cols:
        num_cols = num_cols.drop(target_col)
        
    X = dataframe[num_cols]

    if X.isnull().values.any():
        print(f"NaN/inf found, row will be dropped before VIF calculation.\n")
        X = X.dropna()
    
    X_const = add_constant(X)

    vif = pd.DataFrame({
        'feature': X_const.columns,
        'VIF': [
            variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])
        ]
    })
    vif['VIF'] = vif['VIF'].apply(lambda x: f"{x:,.4f}")

    print(vif.to_string(index=False))


# Input-Output Split
# Function to split data into input features and output target
def split_input_output(data, target_col):
    """
    Splits the data into input features (X) and output target (y).

    Parameters:
    data -- Pandas DataFrame containing the data
    target_col -- String name of the target column to be used as output

    Returns:
    X -- DataFrame containing input features
    y -- Series containing output target
    """
    # Drop the target column to get input features
    X = data.drop(target_col, axis=1)
    
    # Select the target column as output
    y = data[target_col]
    
    # Print the shape of input features DataFrame
    print(X.shape)
    
    # Print the shape of output target Series
    print(y.shape)
    
    # Return input features and output target
    return X, y


def split_train_test(X, y, test_size, stratify, seed):
    """
    Splits the data into training and testing sets.

    Parameters:
    X -- DataFrame containing input features
    y -- Series containing output target
    test_size -- Float representing the proportion of data to include in the test split (e.g., 0.2 for 20%)
    stratify -- Boolean indicating whether to stratify the split according to class labels in y
    seed -- Integer representing the random seed for reproducibility

    Returns:
    X_train -- Training set of input features
    X_test -- Testing set of input features
    y_train -- Training set of output target
    y_test -- Testing set of output target
    """
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=seed
    )
    # Return the split data
    return X_train, X_test, y_train, y_test