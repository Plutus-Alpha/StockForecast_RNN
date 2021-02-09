import pandas as pd
import numpy as np

def moving_average(df, window_size):
    df['{}ma'.format(window_size)] = df['Price'].rolling(window=window_size, min_periods=window_size).mean()

    return df

# type of moving average (MA) that places a greater weight and significance on the most recent data points
def EMA(df, ema_val):
    df['ema{}'.format(ema_val)] = df['Price'].ewm(span=ema_val, min_periods=ema_val).mean()

    return df

# a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price
def MACD(df, ema1, ema2, signal):

    df['EMA{}'.format(ema1)] = df['Price'].ewm(span=ema1, min_periods=ema1).mean()
    df['EMA{}'.format(ema2)] = df['Price'].ewm(span=ema2, min_periods=ema2).mean()

    df['Difference'] = df['EMA{}'.format(ema1)] - df['EMA{}'.format(ema2)]
    df['Signal_{}'.format(signal, ema1, ema2)] = df['Difference'].ewm(span=signal, min_periods=signal).mean()

    df['MACD_{},{},{}'.format(ema1, ema2, signal)] = df['Difference'] - df['Signal_{}'.format(signal)]
    df.drop(columns=['Difference'], inplace=True)
    df.drop(columns=['Signal_{}'.format(signal)], inplace=True)
    df.drop(columns=['EMA{}'.format(ema1)], inplace=True)
    df.drop(columns=['EMA{}'.format(ema2)], inplace=True)

    return df


# Momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time
# Reading above 80 is overbought and below 20 is oversold.
def stochastic_oscillator(df, window):

    low = high = window

    df['L{}'.format(low)] = df['Low'].rolling(window=low, min_periods=low).min()
    df['H{}'.format(high)] = df['High'].rolling(window=high, min_periods=high).max()

    # %K is referred to sometimes as the slow stochastic indicator.
    # The "fast" stochastic indicator is taken as %D = 3-period moving average of %K
    df['%K_W{}'.format(window)] = ((df['Price']-df['L{}'.format(low)])/(df['H{}'.format(high)] - df['L{}'.format(low)]))
    df['%D_W{}'.format(window)] = df['%K_W{}'.format(window)].rolling(window=3, min_periods=3).mean()

    df.drop(columns=['L{}'.format(low)], inplace=True)
    df.drop(columns=['H{}'.format(high)], inplace=True)

    return df


# Williams %R compares the most recent closing price to the highest high of a specified look-back period.
# A reading above -20 is overbought. A reading below -80 is oversold.
# Overbought simply means the price is near the highs of its recent range,
# and oversold means the price is in the lower end of its recent range.
# Overbought readings actually help confirm an uptrend, since a strong uptrend should regularly
# see prices that are pushing to or past prior highs.
def williams_R(df, window):

    low = high = window

    df['L{}'.format(low)] = df['Low'].rolling(window=low, min_periods=low).min()
    df['H{}'.format(high)] = df['High'].rolling(window=high, min_periods=high).max()

    # %K is referred to sometimes as the slow stochastic indicator.
    # The "fast" stochastic indicator is taken as %D = 3-period moving average of %K
    df['%R W{}'.format(window)] = ((df['H{}'.format(high)]-df['Price'])/(df['H{}'.format(high)] - df['L{}'.format(low)]))

    df.drop(columns=['L{}'.format(low)], inplace=True)
    df.drop(columns=['H{}'.format(high)], inplace=True)

    return df


# RSI measures the consistency with which prices increase or decrease over time, so a high RSI reading indicates that
# price has increased with greater frequency than it has declined over a particular time frame.
def relative_strength_index(df, n):
    df.reset_index(inplace=True)

    gain_list = [np.nan]
    loss_list = [np.nan]

    for i in range(1, df.shape[0]):
        if df['Price'].iloc[i] > df['Price'].iloc[i - 1]:
            gain = df['Price'].iloc[i] - df['Price'].iloc[i - 1]
            gain_list.append(gain)
            loss_list.append(0)
        elif df['Price'].iloc[i] == df['Price'].iloc[i - 1]:
            gain_list.append(0)
            loss_list.append(0)
        else:
            loss = abs(df['Price'].iloc[i] - df['Price'].iloc[i - 1])
            loss_list.append(loss)
            gain_list.append(0)

    data = {'gain': gain_list, 'loss': loss_list}
    new_df = pd.DataFrame(data)
    df = df.join(new_df)

    avg_gain_list = [np.nan for _ in range(n)]
    avg_loss_list = [np.nan for _ in range(n)]
    RS_list = [np.nan for _ in range(n)]
    RSI_list = [np.nan for _ in range(n)]

    first_avg_gain = df['gain'][1:n + 1].mean()
    first_avg_loss = df['loss'][1:n + 1].mean()

    avg_gain_list.append(round(first_avg_gain, 6))
    avg_loss_list.append(round(first_avg_loss, 6))
    RS_list.append(avg_gain_list[n] / avg_loss_list[n])
    RSI_list.append(round(100 - (100 / (1 + RS_list[-1])), 3))

    for i in range(n + 1, df.shape[0]):
        avg_gain_list.append(round((avg_gain_list[i - 1] * (n-1) + gain_list[i]) / n, 6))
        avg_loss_list.append(round((avg_loss_list[i - 1] * (n-1) + loss_list[i]) / n, 6))
        RS_list.append(avg_gain_list[i] / avg_loss_list[i])
        RSI_list.append(round(100 - (100 / (1 + RS_list[i])), 3))

    data = {'avg_gain': avg_gain_list, 'avg_loss': avg_loss_list, 'RS': RS_list, 'RSI {}'.format(n): RSI_list}
    new_df = pd.DataFrame(data)
    df = df.join(new_df)

    df.drop(columns=['gain', 'loss', 'avg_gain', 'avg_loss', 'RS'], inplace=True)
    df.set_index('Date', inplace=True)

    return df


# the rate of acceleration of a security's price or volume – that is, the speed at which the price is changing
def momentum(df, n):
    df['momentum {}'.format(n)] = df['Price'] - df['Price'].shift(n)
    return df


def change_rate(df, n):
    df['change_rate {}'.format(n)] = (df['Price']-df['Price'].shift(n)) / df['Price'].shift(n)
    return df


# True range to measure volatility
# A stock experiencing a high level of volatility has a higher ATR, and a low volatility stock has a lower ATR
def average_true_range(df, n):
    df.reset_index(inplace=True)

    df['H-L'] = df['High'] - df['Low']
    df['|H-Cp|'] = abs(df['High'] - df['Price'].shift(1))
    df['|L-Cp|'] = abs(df['Low'] - df['Price'].shift(1))
    # axis = 1 means rowwise max
    df['TR'] = df[['H-L', '|H-Cp|', '|L-Cp|']].max(axis=1)

    ATR_list = [np.nan for _ in range(n)]

    first_mean = df['TR'][1:n + 1].mean()
    ATR_list.append(round(first_mean, 5))

    for i in range(n + 1, df.shape[0]):
        ATR_list.append(round((ATR_list[i - 1] * (n - 1) + df['TR'].iloc[i]) / n, 6))

    data = {'ATR {}'.format(n): ATR_list}
    new_df = pd.DataFrame(data)
    df = df.join(new_df)

    df.drop(columns=['H-L', '|H-Cp|', '|L-Cp|', 'TR'], inplace=True)
    df.set_index('Date', inplace=True)

    max_ATR = df['ATR {}'.format(n)].max()
    min_ATR = df['ATR {}'.format(n)].min()
    ATR_range = max_ATR - min_ATR

    df['ATR_scaled {}'.format(n)] = (df['ATR {}'.format(n)] - min_ATR) / ATR_range

    return df


# ***Must compute True Range before doing this function.
# Used to quantify trend strength.
# The trend has strength when ADX is above 25.
# The trend is weak or the price is trendless when ADX is below 20, according to Wilder.
# The price is moving up when +DI is above -DI, and the price is moving down when -DI is above +DI.
def average_directional_index_and_DI(df, n):
    df.reset_index(inplace=True)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    first_sum_PDM = df['+DM'][1:n + 1].sum()
    first_sum_NDM = df['-DM'][1:n + 1].sum()

    PDM_sum_list = [np.nan for _ in range(n)]
    NDM_sum_list = [np.nan for _ in range(n)]
    TR_list = [np.nan for _ in range(n)]

    PDM_sum_list.append(first_sum_PDM)
    NDM_sum_list.append(first_sum_NDM)

    # Compute smoothed TR
    df['H-L'] = df['High'] - df['Low']
    df['|H-Cp|'] = abs(df['High'] - df['Price'].shift(1))
    df['|L-Cp|'] = abs(df['Low'] - df['Price'].shift(1))
    # axis = 1 means rowwise max
    df['TR'] = df[['H-L', '|H-Cp|', '|L-Cp|']].max(axis=1)

    first_sum = df['TR'][0:n].sum()
    TR_list.append(round(first_sum, 5))

    for i in range(n + 1, df.shape[0]):
        PDM_sum_list.append(round(PDM_sum_list[i - 1] * (n - 1) / n + df['+DM'].iloc[i], 6))
        NDM_sum_list.append(round(NDM_sum_list[i - 1] * (n - 1) / n + df['-DM'].iloc[i], 6))
        TR_list.append(round((TR_list[i - 1] * (n - 1) / n + df['TR'].iloc[i]), 6))

    data = {'+DM {}'.format(n): PDM_sum_list, '-DM {}'.format(n): NDM_sum_list, 'TR {}'.format(n): TR_list}
    new_df = pd.DataFrame(data)
    df = df.join(new_df)

    df['+DI {}'.format(n)] = df['+DM {}'.format(n)] / df['TR {}'.format(n)]
    df['-DI {}'.format(n)] = df['-DM {}'.format(n)] / df['TR {}'.format(n)]

    df['DX {}'.format(n)] = abs((df['+DI {}'.format(n)] - df['-DI {}'.format(n)])) / (
    df['+DI {}'.format(n)] + df['-DI {}'.format(n)])

    ADX_list = [np.nan for _ in range(n * 2 - 1)]
    first_ADX_mean = df['DX {}'.format(n)][n:n * 2].mean()
    ADX_list.append(first_ADX_mean)

    for i in range(n * 2, df.shape[0]):
        ADX_list.append(round((ADX_list[i - 1] * (n - 1) + df['DX {}'.format(n)].iloc[i]) / n, 6))

    data = {'ADX {}'.format(n): ADX_list}
    new_df = pd.DataFrame(data)
    df = df.join(new_df)

    df.drop(columns=['H-L', '|H-Cp|', '|L-Cp|', '+DM', '-DM', 'TR', '+DM {}'.format(n),
                     '-DM {}'.format(n), 'TR {}'.format(n), 'up_move', 'down_move', 'DX {}'.format(n)], inplace=True)
    df.set_index('Date', inplace=True)

    return df