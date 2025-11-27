import gradio as gr
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import os

# --- 1. INISIALISASI SPARK ---
spark = SparkSession.builder \
    .appName("StuntingDeploy") \
    .master("local[1]") \
    .config("spark.driver.memory", "512m") \
    .getOrCreate()

# --- 2. LOAD MODEL ---
MODEL_PATH = "model_rf_stunting"

print("Sedang memuat model...")
try:
    model = PipelineModel.load(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    model = None
    print(f"❌ Gagal memuat model: {e}")

# --- 3. FUNGSI PREDIKSI ---
def predict_stunting(umur, tinggi, jk):
    if model is None:
        return "Error: Model tidak ditemukan."
    
    # Normalisasi Input (Huruf kecil)
    jk_fixed = jk.lower()
    
    # Buat DataFrame
    data = pd.DataFrame({
        'Umur': [float(umur)],
        'Tinggi': [float(tinggi)],
        'JK': [jk_fixed],
        'Status': ['normal'] # Dummy value
    })
    
    spark_df = spark.createDataFrame(data)
    
    try:
        # Prediksi
        result = model.transform(spark_df)
        
        # Ambil hasil prediksi (Angka Index)
        pred_idx = result.select("prediction").collect()[0][0]
        
        # ==================================================================
        # MAPPING HASIL (SESUAI GAMBAR ANDA)
        # ==================================================================
        # Index 0.0 = normal
        # Index 1.0 = tinggi
        # Index 2.0 = severely stunted
        # Index 3.0 = stunted
        
        labels_map = {
            0.0: "✅ Normal (Gizi Baik)",
            1.0: "📏 Tinggi (Perawakan Tinggi)",
            2.0: "⚠️ Sangat Pendek (Severely Stunted)",
            3.0: "⚠️ Pendek (Stunted)"
        }
        
        # Ambil teks status berdasarkan index
        status_teks = labels_map.get(pred_idx, f"Tidak Diketahui (Index: {pred_idx})")
        
        # Logika Saran Sederhana
        saran = ""
        if "Sangat Pendek" in status_teks or "Pendek" in status_teks:
            saran = "\n\nSaran: 🩺 Segera konsultasikan ke Puskesmas, Posyandu, atau Dokter Anak untuk penanganan gizi lebih lanjut."
        elif "Normal" in status_teks:
            saran = "\n\nSaran: 👍 Pertahankan asupan gizi seimbang (protein hewani) dan pantau terus tumbuh kembang anak."
        elif "Tinggi" in status_teks:
            saran = "\n\nSaran: 👌 Pertumbuhan tinggi badan anak sangat baik (di atas rata-rata). Tetap jaga asupan nutrisi."
            
        return status_teks + saran
        
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# --- 4. TAMPILAN UI (GRADIO) ---
ui = gr.Interface(
    fn=predict_stunting,
    inputs=[
        gr.Number(label="Umur (bulan)", value=12),
        gr.Number(label="Tinggi Badan (cm)", value=75),
        gr.Dropdown(["Laki-laki", "Perempuan"], label="Jenis Kelamin", value="Laki-laki")
    ],
    outputs=gr.Textbox(label="Hasil Analisis Status Gizi"),
    title="Sistem Deteksi Dini Stunting 👶",
    description="Masukkan data balita untuk mengetahui status gizi. Sistem menggunakan AI (Random Forest) untuk analisis.",
    theme="soft"
)

if __name__ == "__main__":
    ui.launch()