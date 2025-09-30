import numpy as np
import matplotlib.pyplot as plt

# === Load Price Data ===
with open('pricesv2.txt', 'r') as f:
    lines = f.readlines()

# Convert to 2D numpy array: prices[day, stock]
prices = np.array([[float(x) for x in line.strip().split()] for line in lines]).T
prices = prices.T
num_days, num_stocks = prices.shape

# === Moving Average Parameters ===
short_window = 5
long_window = 70
epsilon = 1e-6  # Small margin to avoid floating point equality issues

# === Generate Trend Following Signals ===
signals = np.zeros_like(prices)

for i in range(num_stocks):  # For each stock
    for day in range(long_window, num_days):
        short_ma = np.mean(prices[day - short_window:day, i])
        long_ma  = np.mean(prices[day - long_window:day, i])

        if short_ma > long_ma + epsilon:
            signals[day, i] = 1 # Buy signal
        elif short_ma < long_ma - epsilon:
            signals[day, i] = -1  # Short signal
        else:
            signals[day, i] = 0  # No action

# === Confirm Signals Are Being Generated ===
print("Number of buy signals:", np.sum(signals == 1))
print("Number of short signals:", np.sum(signals == -1))


# === Initialize Portfolio ===
initial_capital = 10000
portfolio_value = np.zeros(num_days)
portfolio_value[0] = initial_capital
positions = np.zeros(num_stocks)
cash = initial_capital
commission_rate = 0.005


# Optional trade tracking
trade_records = {i: {'positions': [], 'entry_prices': [], 'entry_days': []} for i in range(num_stocks)}

# === Simulate Trading ===
for day in range(1, num_days):
    daily_portfolio_value = cash

    # Add value of open positions
    for i in range(num_stocks):
        daily_portfolio_value += positions[i] * prices[day, i]

    # Execute signals
    for i in range(num_stocks):
        if signals[day, i] == 1 and positions[i] == 0:
            # Enter long
            num_longs = np.sum(signals[day] == 1)
            if num_longs > 0:
                position_value = daily_portfolio_value / num_longs
                shares = position_value / prices[day, i]
                commission = position_value * commission_rate
                cash -= position_value + commission
                positions[i] = shares
                trade_records[i]['positions'].append(shares)
                trade_records[i]['entry_prices'].append(prices[day, i])
                trade_records[i]['entry_days'].append(day)

        elif signals[day, i] == -1 and positions[i] == 0:
            # Enter short
            num_shorts = np.sum(signals[day] == -1)
            if num_shorts > 0:
                position_value = daily_portfolio_value / num_shorts
                shares = position_value / prices[day, i]
                commission = position_value * commission_rate
                cash += position_value - commission  # Cash in from short sale
                positions[i] = -shares
                trade_records[i]['positions'].append(-shares)
                trade_records[i]['entry_prices'].append(prices[day, i])
                trade_records[i]['entry_days'].append(day)

        elif signals[day, i] == 0 and positions[i] > 0:
            # Close long
            position_value = positions[i] * prices[day, i]
            commission = position_value * commission_rate
            cash += position_value - commission
            positions[i] = 0

        elif signals[day, i] == 0 and positions[i] < 0:
            # Cover short
            position_value = -positions[i] * prices[day, i]
            commission = position_value * commission_rate
            cash -= position_value + commission
            positions[i] = 0

    # Update portfolio value
    portfolio_value[day] = cash
    for i in range(num_stocks):
        portfolio_value[day] += positions[i] * prices[day, i]

# === Final Results ===
final_value = portfolio_value[-1]
profit = final_value - initial_capital
profit_pct = (profit / initial_capital) * 100

# === Performance Metrics: Sharpe Ratio & Alpha ===
daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
risk_free_rate = 0.01 / 252  # 1% annual risk-free rate, dailyized

# Sharpe Ratio
excess_daily_returns = daily_returns - risk_free_rate
sharpe_ratio = np.mean(excess_daily_returns) / (np.std(excess_daily_returns) + 1e-8) * np.sqrt(252)

# Alpha (vs. market, here using average stock as proxy)
market_returns = np.mean(np.diff(prices, axis=0) / prices[:-1], axis=1)
market_return_annual = np.mean(market_returns) * 252
strategy_return_annual = np.mean(daily_returns) * 252
beta = np.cov(daily_returns, market_returns)[0, 1] / (np.var(market_returns) + 1e-8)
alpha = (strategy_return_annual - risk_free_rate * 252) - beta * (market_return_annual - risk_free_rate * 252)

# === Plot Portfolio Value ===
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label='Portfolio Value')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
plt.title(f'Trend Following Portfolio Value\nFinal: ${final_value:,.2f} (Profit: ${profit:,.2f}, {profit_pct:.2f}%)')
plt.xlabel('Day')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Prices shape:", prices.shape)  # Should be (750, 50)
print("Sample stock 0:", prices[:10, 0])  # First 10 days of stock 0
print("Min:", np.min(prices), "Max:", np.max(prices))


# === Summary ===
print("\n=== Trend Following Strategy Summary ===")
print(f"Initial Capital:        ${initial_capital:,.2f}")
print(f"Final Portfolio Value:  ${final_value:,.2f}")
print(f"Total Profit:           ${profit:,.2f}")
print(f"Return:                 {profit_pct:.2f}%")
print(f"Number of Trading Days: {num_days}")
print(f"Annualized Return:      {(final_value / initial_capital) ** (252 / num_days) - 1:.2%}")
print(f"Sharpe Ratio:           {sharpe_ratio:.3f}")
print(f"Alpha (annualized):     {alpha:.3f}")