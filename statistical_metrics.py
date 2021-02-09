def normalised_and_root_mean_square_error(df, actual, predict):
    sum_square_error = 0.0

    for i in range(df.shape[0]):
        sum_square_error += (df[actual].iloc[i] - df[predict].iloc[i]) ** 2

    rmse = (sum_square_error / df.shape[0]) ** 0.5
    nrmse = rmse / (df[actual].max() - df[actual].min())

    return round(rmse, 5), round(nrmse, 5)


# Problem: relative size of the error is not always obvious, hard to tell a big error from a small error
def mean_absolute_error(df, actual, predict):
    total_error = 0.0

    for i in range(df.shape[0]):
        total_error += abs(df[actual].iloc[i] - df[predict].iloc[i])

    mae = total_error / df.shape[0]

    return round(mae, 5)


def mean_absolute_percentage_error(df, actual, predict):
    current_percent_error = 0.0

    for i in range(df.shape[0]):
        current_percent_error += abs(df[actual].iloc[i] - df[predict].iloc[i]) / df[actual].iloc[i]

    mape = current_percent_error / df.shape[0] * 100

    return round(mape, 5)


# statistical measure of a model's performance in predicting the direction of change, positive
# or negative, of a time series from one time period to the next.
def directional_symmetry(df, actual, predict):
    d = 0

    for i in range(1, df.shape[0]):
        if (df[actual].iloc[i] - df[actual].iloc[i - 1]) * (df[predict].iloc[i] - df[predict].iloc[i - 1]) > 0:
            d += 1

    ds = d / (df.shape[0] - 1) * 100

    return ds



