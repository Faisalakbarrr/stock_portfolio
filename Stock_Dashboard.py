import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("Dashboard Portofolio Saham")

st.subheader("Identifikasi Tujuan Pencarian Data")
st.write("Melakukan analisis portofolio investasi 5 perusahaan BUMN yang terdaftar di Bursa Efek Indonesia.")

st.subheader("Sumber Data Eksternal")
st.write("yahoo finance, id.investing.com dan Bursa Efek Indonesia (idx.co.id)")

st.subheader("Pencarian Data")
st.write("Mengunjungi situs web id.investing.com, Bursa Efek Indonesia dan yahoo finance, mengakses bagian yang menyediakan data historis harga saham closing untuk mendapatkan data harga saham closing yang diinginkan.")


# Daftar simbol saham
saham = ['KAEF.JK', 'PGAS.JK', 'PTBA.JK', 'SMBR.JK', 'TLKM.JK']

# Tanggal awal dan akhir
start_date = '2021-01-04'
end_date = '2022-12-30'

# Mengambil data harga closing saham untuk setiap perusahaan
data = yf.download(saham, start=start_date, end=end_date)['Close']

# Menampilkan data harga closing saham
st.subheader("Data Harga Closing Saham")
st.dataframe(data)

# Menampilkan deskripsi statistik
st.subheader("Deskripsi Statistik")
st.write(data.describe())

# Menampilkan grafik harga closing saham
st.subheader("Grafik Harga Closing Saham")
fig = px.line(data, labels={"x": "Tanggal", "y": "Harga Closing Saham"})
st.plotly_chart(fig)

st.subheader("Penjelasan")
st.write("Plot tersebut menampilkan perubahan harga penutupan dari waktu ke waktu untuk setiap perusahaan saham yang dianalisis. Sumbu x adalah rentang tanggal dari 4 Januari 2021 hingga 30 Desember 2022, sedangkan sumbu y adalah harga penutupan saham untuk masing-masing perusahaan.")

st.write("1. Dalam grafik ini, terlihat bahwa saham PGAS, PTBA, dan TLKM mengalami tren kenaikan yang signifikan. Selain itu, grafik untuk saham-saham ini memiliki fluktuasi yang lebih kecil dan pergerakan yang lebih stabil, yang menunjukkan tingkat volatilitas yang lebih rendah.")

st.write("2. Di sisi lain, saham KAEF dan SMBR mengalami penurunan. Terutama pada saham KAEF, terlihat penurunan yang signifikan.")

st.write("Secara umum, plot ini membantu mengidentifikasi tren dalam harga saham setiap perusahaan. Informasi ini dapat membantu dalam pengambilan keputusan investasi atau strategi perdagangan.")


# Interaksi untuk memilih saham yang ditampilkan
selected_stocks = st.multiselect("Pilih Saham", data.columns)

# Filter data berdasarkan saham yang dipilih
selected_data = data[selected_stocks]

# Menampilkan grafik harga closing saham yang dipilih
st.subheader("Grafik Harga Closing Saham Terpilih")
fig_selected = px.line(selected_data, labels={"x": "Tanggal", "y": "Harga Closing Saham"})
st.plotly_chart(fig_selected)

# Matriks Korelasi
st.subheader("Matriks Korelasi")
correlation = data.corr()

# Visualisasi Heatmap dari Matriks Korelasi
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
ax.set_title("Matriks Korelasi")
st.pyplot(fig)

st.subheader("Matriks Korelasi")
st.write("Dengan menggunakan matriks korelasi, kita dapat melihat hubungan antara harga penutupan saham dari setiap perusahaan dalam portofolio Anda.")

st.write("1. Terlihat bahwa saham-saham SMBR dan KAEF, PTBA dan PGAS, TLKM dan PGAS, serta TLKM dan PTBA memiliki korelasi yang sangat positif. Ini menunjukkan bahwa perusahaan-perusahaan ini cenderung bergerak dalam arah yang sama.")

st.write("2. Di sisi lain, saham-saham PGAS dan KAEF, PTBA dan KAEF, TLKM dan KAEF, TLKM dan PGAS, serta SMBR dan TLKM memiliki korelasi yang sangat negatif. Ini menunjukkan bahwa perusahaan-perusahaan ini cenderung bergerak dalam arah yang berlawanan.")


# Risk & Return
returns = data.pct_change().dropna()

# Plotting Daily Simple Returns
st.subheader('Daily Returns')
fig_returns, ax_returns = plt.subplots(figsize=(10, 5))

for i in returns.columns.values:
    ax_returns.plot(returns[i], lw=2, label=i)

ax_returns.legend(loc='upper right', fontsize=10)
ax_returns.set_title('Volatility in Daily Returns')
ax_returns.set_xlabel('Date')
ax_returns.set_ylabel('Daily Returns')
st.pyplot(fig_returns)

st.write("Berdasarkan grafik di samping, secara harian, kaef pada umumnya adalah yang paling fluktuatif dibandingkan saham-saham individual mana pun.")

# Menghitung Rata-rata Daily Simple Returns
average_returns = returns.mean()

# Menampilkan Rata-rata Daily Simple Returns
st.subheader('Average Daily Returns')
st.bar_chart(average_returns)

st.subheader('Penjelasan')
st.write("Rata-rata Daily Simple Returns:")
st.write("- KAEF: Rata-rata Daily Simple Returns untuk KAEF adalah -0.002205.")
st.write("- SMBR: Rata-rata Daily Simple Returns untuk SMBR adalah -0.001656.")
st.write("  Hal ini mengindikasikan adanya penurunan rata-rata dalam harga saham KAEF & SMBR pada setiap hari perdagangan.")
st.write("- PGAS: Rata-rata Daily Simple Returns untuk PGAS adalah 0.000572.")
st.write("- PTBA: Rata-rata Daily Simple Returns untuk PTBA adalah 0.000819.")
st.write("- TLKM: Rata-rata Daily Simple Returns untuk TLKM adalah 0.000293.")
st.write("  Hal ini mengindikasikan adanya peningkatan rata-rata dalam harga saham PGAS, PTBA, dan TLKM pada setiap hari perdagangan.")

st.write("Analisis ini dapat memberikan gambaran tentang kinerja relatif perusahaan-perusahaan dalam portofolio saham Anda. Penting untuk diingat bahwa rata-rata harian hanya memberikan gambaran singkat tentang perubahan harga saham.")


# Menghitung Annualized Standard Deviation
annualized_std = returns.std() * np.sqrt(252) * 100

# Menampilkan Annualized Standard Deviation
st.subheader('Annualized Standard Deviation (Volatility, 493 trading days)')
st.bar_chart(annualized_std)

# Menampilkan Annualized Standard Deviation
st.subheader('Penjelasan')
st.write("Analisis Annualized Standard Deviation mengukur tingkat fluktuasi atau volatilitas harga saham dalam satu tahun perdagangan.")

# Menampilkan tabel Annualized Standard Deviation
st.write("Annualized Standard Deviation:")
st.write("- KAEF: Annualized Standard Deviation untuk saham KAEF adalah 56.818076. Tingkat volatilitas yang tinggi dalam harga saham KAEF menunjukkan fluktuasi besar dalam periode perdagangan satu tahun, sehingga berisiko tinggi.")
st.write("- SMBR: Annualized Standard Deviation untuk saham SMBR adalah 56.818076. Tingkat volatilitas yang tinggi dalam harga saham SMBR menunjukkan fluktuasi besar dalam periode perdagangan satu tahun, sehingga berisiko tinggi.")
st.write("- PGAS: Annualized Standard Deviation untuk saham PGAS adalah 40.431764. Tingkat volatilitas yang cukup tinggi dalam harga saham PGAS menunjukkan fluktuasi yang signifikan dalam periode perdagangan satu tahun.")
st.write("- PTBA: Annualized Standard Deviation untuk saham PTBA adalah 42.691419. Tingkat volatilitas yang cukup tinggi dalam harga saham PTBA menunjukkan fluktuasi yang signifikan dalam periode perdagangan satu tahun.")
st.write("- TLKM: Annualized Standard Deviation untuk saham TLKM adalah 27.357722. Tingkat volatilitas yang lebih rendah dalam harga saham TLKM dibandingkan dengan saham lainnya dalam portofolio, dengan fluktuasi yang lebih kecil dalam periode perdagangan satu tahun.")

st.write("Analisis ini membantu dalam mengevaluasi tingkat risiko dan volatilitas masing-masing saham dalam portofolio Anda.")



# Menghitung Daily Cumulative Simple Returns
daily_cumulative_simple_return = (returns + 1).cumprod()

# Menampilkan Daily Cumulative Simple Returns
st.subheader('Daily Cumulative Simple Returns')
st.dataframe(daily_cumulative_simple_return)

# Visualisasi Daily Cumulative Simple Returns
st.subheader('Visualization of Daily Cumulative Simple Returns')
fig_cumulative, ax_cumulative = plt.subplots(figsize=(10, 5))

for i in daily_cumulative_simple_return.columns.values:
    ax_cumulative.plot(daily_cumulative_simple_return[i], lw=2, label=i)

ax_cumulative.legend(loc='upper right', fontsize=10)
ax_cumulative.set_title('Daily Cumulative Simple Returns / Growth of Investment')
ax_cumulative.set_xlabel('Date')
ax_cumulative.set_ylabel('Growth of ₨ 1 investment')
st.pyplot(fig_cumulative)

# Menampilkan Daily Cumulative Simple Returns
st.subheader('Daily Cumulative Simple Returns')
st.write("Analisis Daily Cumulative Simple Returns menggambarkan pertumbuhan investasi jika kita menginvestasikan 1 unit pada awal periode dan mempertahankan investasi tersebut.")

# Menampilkan grafik Daily Cumulative Simple Returns
st.write("Grafik Daily Cumulative Simple Returns:")
# Kode untuk menampilkan grafik di sini

# Menampilkan analisis berdasarkan grafik
st.write("Berdasarkan grafik di atas, selama rentang 2 tahun dari 2021-2022:")
st.write("- PTBA menunjukkan kinerja terbaik dan menghasilkan pengembalian paling kumulatif.")
st.write("- Diikuti oleh PGAS, dan di urutan ketiga, TLKM. Hal ini menunjukkan bahwa investasi menghasilkan keuntungan dan memiliki kinerja yang lebih baik serta pertumbuhan investasi yang cukup besar.")
st.write("- KAEF dan SMBR memiliki pengembalian kumulatif menurun selama periode 2 tahun, dengan garis pertumbuhan yang lebih rendah menunjukkan kinerja yang lebih lemah dan pertumbuhan yang lebih kecil dari investasi. Hal ini menunjukkan bahwa investasi mengalami kerugian.")

st.write("Analisis ini membantu dalam mengevaluasi pertumbuhan dan kinerja investasi dalam jangka waktu tertentu.")

