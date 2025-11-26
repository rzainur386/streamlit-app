import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ============================
# Konfigurasi Halaman
# ============================
st.set_page_config(page_title="Aplikasi Loyalitas Pelanggan", layout="wide")

# ============================
# Navigasi Sidebar
# ============================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "K-Means Clustering"])

# ============================
# HALAMAN BERANDA
# ============================
if page == "Beranda":
    st.title("Aplikasi Algoritma K-Means Clustering Untuk Analisis Loyalitas Pelanggan Pada Konveksi Al-Maidah Situbondo")

    st.markdown("Aplikasi ini bertujuan untuk mengetahui tingkat loyalitas pelanggan berdasarkan data pembelian di Konveksi Al-Maidah Situbondo.")

    st.image("sss.png", use_container_width=True)

    st.markdown("""
        ### Fitur Aplikasi:
        - Upload data pelanggan dalam format CSV
        - Penentuan jumlah cluster dengan Elbow Method
        - Evaluasi cluster menggunakan Davies-Bouldin Index
        - Normalisasi data agar skala variabel seimbang
        - Visualisasi hasil clustering dan kategori loyalitas
        - Perbandingan grafik sebelum & sesudah normalisasi
        - Download hasil klasifikasi dalam format CSV
    """)

# ============================
# HALAMAN CLUSTERING
# ============================
elif page == "K-Means Clustering":
    st.title("📊 K-Means Clustering Loyalitas Pelanggan")

    uploaded_file = st.file_uploader("📂 Upload dataset (.csv)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=";")
        st.subheader("📊 Data Awal")
        st.dataframe(df)

        # Pilih kolom untuk clustering
        selected_columns = st.multiselect(
            "🧮 Pilih kolom untuk clustering",
            df.columns.tolist(),
            default=["JumlahPesanan", "TotalHarga", "Frekuensi"]
        )

        if len(selected_columns) >= 2:
            X_raw = df[selected_columns]

            # ============================
            # Normalisasi Data
            # ============================
            st.subheader("⚖️ Normalisasi Data")
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X_raw)
            X = pd.DataFrame(X_normalized, columns=selected_columns)
            st.dataframe(X)

            # ============================
            # Grafik Perbandingan
            # ============================
            st.subheader("📊 Perbandingan Sebelum & Sesudah Normalisasi")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Grafik sebelum normalisasi
            X_raw.plot(kind='box', ax=axes[0])
            axes[0].set_title("Sebelum Normalisasi")
            axes[0].set_ylabel("Skala Asli")

            # Grafik sesudah normalisasi
            X.plot(kind='box', ax=axes[1])
            axes[1].set_title("Sesudah Normalisasi")
            axes[1].set_ylabel("Skala 0-1")

            st.pyplot(fig)

            # ============================
            # Elbow Method
            # ============================
            st.subheader("📈 Elbow Method (Menentukan K terbaik)")
            distortions = []
            K = range(1, min(11, len(X)))
            for k in K:
                if k < len(X):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X)
                    distortions.append(kmeans.inertia_)

            fig1, ax1 = plt.subplots()
            ax1.plot(K, distortions, 'bx-')
            ax1.set_xlabel('Jumlah Cluster (K)')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Grafik Elbow')
            st.pyplot(fig1)

            max_k = min(10, len(X) - 1)
            k = st.slider("🔢 Pilih jumlah cluster (K)", min_value=2, max_value=max_k, value=3)

            # ============================
            # K-Means Clustering
            # ============================
            model = KMeans(n_clusters=k, random_state=42)
            cluster_labels = model.fit_predict(X)
            df["Cluster"] = cluster_labels

            # ============================
            # Label Loyalitas
            # ============================
            cluster_means = df.groupby("Cluster")[selected_columns].mean()
            cluster_means["TotalSkor"] = cluster_means.sum(axis=1)
            sorted_clusters = cluster_means.sort_values("TotalSkor", ascending=False).index.tolist()

            cluster_label_map = {}
            for idx, cluster_id in enumerate(sorted_clusters):
                if idx == 0:
                    cluster_label_map[cluster_id] = "Sangat Loyal"
                elif idx == 1:
                    cluster_label_map[cluster_id] = "Cukup Loyal"
                else:
                    cluster_label_map[cluster_id] = "Tidak Loyal"

            df["Kategori Loyalitas"] = df["Cluster"].map(cluster_label_map)

            # ============================
            # Hasil Clustering
            # ============================
            st.subheader("📋 Hasil Clustering")
            st.dataframe(df)

            # ============================
            # Visualisasi Cluster Plot
            # ============================
            if len(selected_columns) == 2:
                st.subheader("📌 Visualisasi Cluster Plot")
                fig2, ax2 = plt.subplots()
                ax2.scatter(
                    X[selected_columns[0]],
                    X[selected_columns[1]],
                    c=cluster_labels,
                    cmap='viridis'
                )
                ax2.set_xlabel(selected_columns[0])
                ax2.set_ylabel(selected_columns[1])
                ax2.set_title("Cluster Plot")
                st.pyplot(fig2)

            # ============================
            # Davies-Bouldin Index (DBI)
            # ============================
            st.subheader("📊 Nilai DBI untuk K = 2 sampai 10")
            dbi_scores = []
            for i in range(2, max_k + 1):
                if i < len(X):
                    km = KMeans(n_clusters=i, random_state=42)
                    labels = km.fit_predict(X)
                    dbi = davies_bouldin_score(X, labels)
                    dbi_scores.append((i, dbi))

            dbi_df = pd.DataFrame(dbi_scores, columns=["Jumlah Cluster (K)", "DBI"])
            st.dataframe(dbi_df)

            selected_dbi = dbi_df[dbi_df["Jumlah Cluster (K)"] == k]["DBI"].values[0]
            st.success(f"🎯 Nilai DBI untuk K={k}: **{selected_dbi:.4f}**")

            # ============================
            # Hasil Klasifikasi Loyalitas
            # ============================
            st.subheader("📌 Hasil Klasifikasi Loyalitas")
            st.dataframe(df[["CustomerID", "Cluster", "Kategori Loyalitas"]])

            # ============================
            # Download hasil
            # ============================
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df(df)
            st.download_button(
                label="📥 Download Hasil Clustering",
                data=csv_data,
                file_name='hasil_cluster.csv',
                mime='text/csv'
            )
        else:
            st.warning("⚠️ Pilih minimal 2 kolom untuk clustering.")


## buat jalan python -m streamlit run app.py

