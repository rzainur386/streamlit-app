import streamlit as st
import pandas as pd
import numpy as np

# ============================
# FUNGSI K-MEANS MANUAL
# ============================
def euclidean_distance(X, centroids):
    dist = np.zeros((X.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        dist[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
    return dist


def kmeans_manual(X, centroids_init, max_iter=100):
    centroids = centroids_init.copy()
    history = []

    for iteration in range(1, max_iter + 1):
        distances = euclidean_distance(X, centroids)
        clusters = np.argmin(distances, axis=1)

        history.append({
            "iteration": iteration,
            "distances": distances.copy(),
            "clusters": clusters.copy(),
            "centroids": centroids.copy()
        })

        # Hitung centroid baru
        new_centroids = []
        for k in range(len(centroids)):
            members = X[clusters == k]
            if len(members) > 0:
                new_centroids.append(members.mean(axis=0))
            else:
                new_centroids.append(centroids[k])

        new_centroids = np.array(new_centroids)

        # STOP jika centroid tidak berubah
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return history, centroids, clusters


# ============================
# KONFIGURASI HALAMAN
# ============================
st.set_page_config(page_title="K-Means Manual", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "K-Means Clustering"])

# ============================
# BERANDA
# ============================
if page == "Beranda":
    st.title("Aplikasi K-Means Clustering Manual")
    st.markdown("""
    Aplikasi ini menampilkan **seluruh proses K-Means Clustering secara manual**
    mulai dari:
    - Centroid awal
    - Perhitungan jarak
    - Iterasi 1, 2, 3, ...
    - Pembaruan centroid
    - Hasil akhir clustering
    """)

# ============================
# HALAMAN CLUSTERING
# ============================
elif page == "K-Means Clustering":
    st.title("📊 K-Means Clustering (Manual – Full Iterasi)")

    uploaded_file = st.file_uploader("📂 Upload dataset (.csv)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=";", decimal=",")
        st.subheader("📊 Data Awal")
        st.dataframe(df)

        selected_columns = st.multiselect(
            "🧮 Pilih kolom untuk clustering",
            df.columns.tolist(),
            default=["JumlahPesanan", "TotalHarga", "Frekuensi"]
        )

        if len(selected_columns) >= 2:
            X = df[selected_columns].to_numpy()

            # Samakan skala seperti Excel (bagi 1000 pada TotalHarga)
            X[:, 1] = X[:, 1] / 1000

            st.subheader("📌 Data Digunakan ")
            st.dataframe(df[selected_columns])

            # ============================
            # CENTROID AWAL
            # ============================
            if len(df) < 150:
                st.error("❌ Data minimal harus 150 baris!")
                st.stop()

            centroids_init = X[[49, 99, 149]]

            st.subheader("🔹 Centroid Awal")
            st.dataframe(pd.DataFrame(
                centroids_init,
                columns=selected_columns,
                index=["C1 (Data 50)", "C2 (Data 100)", "C3 (Data 150)"]
            ))

            # ============================
            # PROSES K-MEANS MANUAL
            # ============================
            history, final_centroids, final_clusters = kmeans_manual(X, centroids_init)

            # ============================
            # TAMPILKAN SEMUA ITERASI
            # ============================
            for step in history:
                st.markdown(f"## 🔁 Iterasi {step['iteration']}")

                dist_df = pd.DataFrame(
                    step["distances"],
                    columns=["Jarak C1", "Jarak C2", "Jarak C3"]
                )
                dist_df["Cluster"] = ["C" + str(c + 1) for c in step["clusters"]]

                tampil_df = pd.concat(
                    [df[["ID Customer"]], dist_df],
                    axis=1
                )

                st.dataframe(tampil_df)

                # JML & RATA-RATA
                st.markdown("### 📊 Rekap Iterasi")
                recap = tampil_df.groupby("Cluster").size().reset_index(name="Jumlah Data")
                st.dataframe(recap)

                centroid_df = pd.DataFrame(
                    step["centroids"],
                    columns=selected_columns,
                    index=["C1", "C2", "C3"]
                )
                st.markdown("### 📌 Centroid Iterasi Ini")
                st.dataframe(centroid_df)

            # ============================
            # HASIL AKHIR
            # ============================
            df["Cluster"] = ["C" + str(c + 1) for c in final_clusters]

            st.subheader("✅ Hasil Akhir Clustering")
            st.dataframe(df)

            # ============================
            # INTERPRETASI LOYALITAS
            # ============================
            st.subheader("🏷️ Keterangan Loyalitas Pelanggan")

            # Hitung rata-rata tiap cluster
            cluster_summary = (
                df.groupby("Cluster")[selected_columns]
                .mean()
            )

            # Total skor (untuk perangkingan)
            cluster_summary["TotalSkor"] = cluster_summary.sum(axis=1)

            st.markdown("### 📊 Rata-rata Setiap Cluster")
            st.dataframe(cluster_summary.drop(columns=["TotalSkor"]))

            # Urutkan dari paling loyal
            ranking = cluster_summary.sort_values("TotalSkor", ascending=False).index.tolist()

            label_map = {
                ranking[0]: "Sangat Loyal",
                ranking[1]: "Cukup Loyal",
                ranking[2]: "Tidak Loyal"
            }

            # Tambahkan ke data
            df["Kategori Loyalitas"] = df["Cluster"].map(label_map)

            st.markdown("### 🧾 Hasil Akhir + Keterangan Loyalitas")
            st.dataframe(df)

            # Rekap jumlah pelanggan per kategori
            st.markdown("### 📌 Rekap Jumlah Pelanggan")
            rekap = df["Kategori Loyalitas"].value_counts().reset_index()
            rekap.columns = ["Kategori Loyalitas", "Jumlah Pelanggan"]
            st.dataframe(rekap)


            # ============================
            # DOWNLOAD
            # ============================
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Hasil",
                csv,
                "hasil_kmeans_manual.csv",
                "text/csv"
            )

        else:
            st.warning("⚠️ Pilih minimal 2 kolom.")





