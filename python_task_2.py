import pandas as pd
import numpy as np
import datetime


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    all_ids = sorted(set(df['id_start']).union(set(df['id_end'])))

    # Create an empty square matrix for distances
    n = len(all_ids)
    distance_array = np.zeros((n, n))

    # Map IDs to indices for easier manipulation
    id_to_index = {id_val: i for i, id_val in enumerate(all_ids)}

    # Fill the distance matrix with cumulative distances
    for _, row in df.iterrows():
        start_idx = id_to_index[row['id_start']]
        end_idx = id_to_index[row['id_end']]
        distance = row['distance']

        # Update the distance if a shorter path is found
        if distance_array[start_idx, end_idx] == 0 or distance_array[start_idx, end_idx] > distance:
            distance_array[start_idx, end_idx] = distance
            distance_array[end_idx, start_idx] = distance  # Make the matrix symmetric

    # Calculate cumulative distances
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_array[i, k] != 0 and distance_array[k, j] != 0:
                    if distance_array[i, j] == 0 or distance_array[i, j] > distance_array[i, k] + distance_array[k, j]:
                        distance_array[i, j] = distance_array[i, k] + distance_array[k, j]

    # Set diagonal values to 0
    np.fill_diagonal(distance_array, 0)

    # Create a DataFrame from the computed distance matrix
    distance_matrix = pd.DataFrame(distance_array, index=all_ids, columns=all_ids)

    df = distance_matrix[::]

    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_distances = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                unrolled_distances.append([id_start, id_end, distance])

    unrolled_df = pd.DataFrame(unrolled_distances, columns=['id_start', 'id_end', 'distance'])
    df = unrolled_df[::]
    return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_distances = df[(df['id_start'] == reference_id) | (df['id_end'] == reference_id)]['distance']
    reference_avg_distance = reference_distances.mean()

    # Calculate the threshold range
    threshold = 0.1 * reference_avg_distance
    lower_bound = reference_avg_distance - threshold
    upper_bound = reference_avg_distance + threshold

    # Find IDs within the threshold range
    within_threshold_ids = []
    for id_val in df['id_start'].unique():
        avg_distance = df[(df['id_start'] == id_val) | (df['id_end'] == id_val)]['distance'].mean()
        if lower_bound <= avg_distance <= upper_bound:
            within_threshold_ids.append(id_val)

    # Sort the IDs within the threshold range
    sorted_within_threshold_ids = sorted(within_threshold_ids)

    df = pd.DataFrame({'id_start': sorted_within_threshold_ids})

    return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type based on distance
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_ranges = [(datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
                   (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
                   (datetime.time(18, 0, 0), datetime.time(23, 59, 59))]

    # Define discount factors for weekdays and weekends
    weekday_discounts = {0: 0.8, 1: 1.2, 2: 0.8}  # Discount factors for different time ranges on weekdays
    weekend_discount = 0.7  # Constant discount factor for weekends

    # Iterate over each (id_start, id_end) pair and time range, covering a full 24-hour period and all 7 days
    all_data = []
    for day_idx, day_name in enumerate(day_names):
        for id_start, id_end in zip(df['id_start'], df['id_end']):
            for start_time, end_time in time_ranges:
                start_datetime = datetime.datetime.combine(datetime.date.today(), start_time)
                end_datetime = datetime.datetime.combine(datetime.date.today(), end_time)
                start_day = day_name
                end_day = day_names[(day_idx + 1) % 7]  # Next day for end_day

                if day_idx < 5:  # Weekdays
                    discount = weekday_discounts[(day_idx + 1) % 3]  # Apply discount based on time range
                else:  # Weekends
                    discount = weekend_discount

                data = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': start_day,
                    'end_day': end_day,
                    'start_time': start_time,
                    'end_time': end_time,
                }

                # Apply the discount factor to each vehicle type column
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    data[vehicle] = df[(df['id_start'] == id_start) &
                                        (df['id_end'] == id_end)][vehicle].values[0] * discount

                all_data.append(data)

    # Create a DataFrame from the collected data
    result_df = pd.DataFrame(all_data)

    df = result_df[::]

    return df