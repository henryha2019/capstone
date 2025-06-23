import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_clean_data(file_path):
    """
    Load and clean a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values removed.
    """
    df = pd.read_csv(file_path)
    cleaned_df = df.dropna()
    return cleaned_df


def show_summary(df):
    """
    Display a summary of a DataFrame, including info, statistical summary,
    and count of missing values.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.

    Returns:
        None
    """
    print("data Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isna().sum())


def plot_feature_distributions(df, features, title, remove_outliers=False, save_path=None):
    """
    Plot histograms of selected features grouped by location, with an option to remove outliers.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list of str): List of feature column names to visualize.
        title (str): Title of the plot.
        remove_outliers (bool, optional): Whether to remove outliers from the plot. Defaults to False.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None
    """
    image_ext = ".png"
    if remove_outliers:
        image_ext = "_rm_outliers" + image_ext
        # Remove outliers based on IQR for each feature
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    locations = df['location'].unique()
    colors = plt.get_cmap('tab10', len(locations))

    cols = 2
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)

    for idx, feature in enumerate(features):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        for i, loc in enumerate(locations):
            subset = df[df['location'] == loc]
            ax.hist(subset[feature].dropna(), bins=30, alpha=0.6,
                    label=loc, color=colors(i), edgecolor='black', linewidth=0.5)
        ax.set_title(feature)
        ax.legend(title='Location', fontsize=8)

    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j // cols][j % cols])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(f"{save_path}feature_distributions{image_ext}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_correlation_heatmap(df, features, title, save_path=None):
    """
    Plot a heatmap showing the correlation matrix of selected features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list of str): List of feature column names to include in the heatmap.
        title (str): Title of the heatmap.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(f"{save_path}feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_boxplots(df, features, title, remove_outliers=False, save_path=None):
    """
    Plot boxplots for selected features grouped by location, with an option to remove outliers.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        features (list of str): List of feature column names to visualize.
        title (str): Title of the plot.
        remove_outliers (bool, optional): Whether to remove outliers from the plot. Defaults to False.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None
    """
    image_ext = ".png"
    if remove_outliers:
        image_ext = "_rm_outliers" + image_ext
        # Remove outliers based on IQR for each feature
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()
    plt.suptitle(title, fontsize=16)

    for i, feature in enumerate(features):
        sns.boxplot(data=df, x="location", y=feature, ax=axes[i])
        axes[i].set_title(feature)
        axes[i].set_xlabel("Location")
        axes[i].set_ylabel(feature)

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(f"{save_path}/feature_boxplots{image_ext}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_target_distributions(df, targets, title, save_path=None):
    """
    Plot histograms of target columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        targets (list of str): List of target column names to visualize.
        title (str): Title of the plot.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None
    """
    df[targets].hist(bins=30, figsize=(12, 8), color="blue", alpha=0.7)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/target_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_target_boxplots(df, targets, title, save_path=None):
    """
    Plot boxplots for target columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        targets (list of str): List of target column names to visualize.
        title (str): Title of the plot.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None
    """
    target_data_long = df[targets].melt(var_name="Target", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=target_data_long, y="Target", x="Value", color="skyblue")
    plt.title(title, fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Target")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/target_boxplots.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_device_name(device):
    """
    Converts a device name to a readable, formatted string.

    - The input string is converted to uppercase.
    - Underscores (_) in the name are replaced with spaces.

    Args:
        device (str): The original device name in raw format (e.g., 'device_1').

    Returns:
        str: A formatted device name (e.g., 'DEVICE 1').
    """
    return device.upper().replace('_', ' ')

def plot_input_target_correlation_heatmap(df, input_features, target_features, title, save_path=None):
    """
    Plot a heatmap showing the correlation between input and target features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        input_features (list of str): List of input feature column names.
        target_features (list of str): List of target feature column names.
        title (str): Title of the heatmap.
        save_path (str, optional): File path to save the plot.

    Returns:
        None
    """
    if not set(input_features + target_features).issubset(df.columns):
        raise ValueError("Some input or target features are missing from the DataFrame.")

    correlation_matrix = df[input_features + target_features].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix.loc[input_features, target_features],
        annot=True, fmt=".2f", cmap="coolwarm", cbar=True
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Target Features")
    plt.ylabel("Input Features")

    if save_path:
        plt.savefig(f"{save_path}/input_target_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_input_target_pairplot(df, input_features, target_features, title,
                                      save_path=None):
    """
    Plot a pairplot showing the relationship between input features and target features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        input_features (list of str): List of input feature column names.
        target_features (list of str): List of target feature column names.
        title (str): Title of the Pairplot.
        save_path (str, optional): File path to save the plot.

    Returns:
        None
    """
    if not set(input_features + target_features).issubset(df.columns):
        raise ValueError("Some input or target features are missing from the DataFrame.")

    sns.pairplot(df, x_vars=input_features, y_vars=target_features, height=3.5)
    plt.suptitle(title, y=1.02, fontsize=16)

    if save_path:
        plt.savefig(f"{save_path}/input_target_pairplot.png", dpi=300, bbox_inches="tight")
    plt.show()