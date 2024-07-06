from tabulate import tabulate

def print_df_in_chunks(df, n):
    """
    Prints the DataFrame in chunks of n columns using tabulate.

    Parameters:
    df (pd.DataFrame): The DataFrame to print.
    n (int): The number of columns per chunk.
    """
    start = 0
    end = n
    total_columns = df.shape[1]
    
    while start < total_columns:
        print(tabulate(df.iloc[:, start:end].head(), headers="keys", tablefmt="orgtbl"))
        start = end
        end += n
        print()  # Add an empty line between chunks for better readability