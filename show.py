import pandas as pd
import matplotlib.pyplot as plt

file_path = 'dataflow.xlsx'
df = pd.read_excel(file_path)
df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df = df.sort_values(by='date')

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['flow'], linestyle='-', marker=None, linewidth=2)
plt.title('Spring Water Flow Line Chart')
plt.xlabel('Date')
plt.ylabel('Spring-Water')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.show()