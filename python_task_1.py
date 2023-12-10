import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix ge
        nerated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    id1_unique = set(df['id_1'].unique())
    id2_unique = set(df['id_2'].unique())

    unique_ids = sorted(id1_unique | id2_unique)

    car_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids, dtype=float)

    for index, row in df.iterrows():
        car_matrix.at[row['id_1'], row['id_2']] = row['car']

    df = car_matrix[::]

    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Count occurrences of each 'car_type'
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = sorted(type_counts.items())

    return dict(sorted_type_counts)


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify the indexes where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indexes in ascending order
    bus_indexes.sort()

    return list(bus_indexes)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_means = df.groupby('route')['truck'].mean()

    filtered_routes = route_means[route_means > 7].index.tolist()

    filtered_routes.sort()

    return list(filtered_routes)


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    modified_matrix = modified_matrix.round(1)

    return modified_matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'],errors = 'coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'],errors = 'coerce')

    # Create a new column 'duration' to store the duration of each timestamp pair
    df['duration'] = df['end_datetime'] - df['start_datetime']

    # Create a mask for pairs with incorrect timestamps
    incorrect_mask = (df['duration'] < pd.Timedelta(days=1)) | (df['duration'] > pd.Timedelta(days=7))

    # Group by (id, id_2) and check if any pair has incorrect timestamps
    result_series = df.groupby(['id', 'id_2'])['duration'].apply(lambda x: any(incorrect_mask[x.index]))

    return pd.Series(result_series)