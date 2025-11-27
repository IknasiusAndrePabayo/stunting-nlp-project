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
# Tetap menggunakan nama folder lama sesuai permintaan
MODEL_PATH = "model_rf_stunting" 

print("Sedang memuat model...")
try:
    model = PipelineModel.load(MODEL_PATH)
    print(f"✅ Model dari '{MODEL_PATH}' berhasil dimuat!")
except Exception as e:
    model = None
    print(f"❌ Gagal memuat model: {e}")

# --- 3. FUNGSI PREDIKSI ---
def predict_stunting(umur, tinggi, jk):
    if model is None:
        return f"Error: Folder '{MODEL_PATH}' tidak ditemukan."
    
    # [VALIDASI] Batasi umur sesuai dataset (0-19 bulan)
    if umur < 0 or umur > 19:
        return "⚠️ Peringatan: Model ini hanya akurat untuk balita usia 0-19 bulan (sesuai dataset latih)."

    # Normalisasi Input
    jk_fixed = jk.lower()
    
    # [FEATURE ENGINEERING] Hitung Rasio Manual
    try:
        umur_float = float(umur)
        tinggi_float = float(tinggi)
        # Rumus Rasio = Tinggi / (Umur + 0.1)
        rasio_calc = tinggi_float / (umur_float + 0.1)
    except ValueError:
        return "Error: Input harus berupa angka."

    # Buat DataFrame (Sertakan kolom 'Rasio')
    data = pd.DataFrame({
        'Umur': [umur_float],
        'Tinggi': [tinggi_float],
        'JK': [jk_fixed],
        'Status': ['normal'], # Dummy
        'Rasio': [rasio_calc] 
    })
    
    spark_df = spark.createDataFrame(data)
    
    try:
        # Prediksi
        result = model.transform(spark_df)
        pred_idx = result.select("prediction").collect()[0][0]
        
        # MAPPING HASIL (Sesuai gambar Colab Anda: 0=Normal, 1=Tinggi, 2=Sev.Stunted, 3=Stunted)
        labels_map = {
            0.0: "✅ Normal (Gizi Baik)",
            1.0: "📏 Tinggi (Perawakan Tinggi)",
            2.0: "⚠️ Sangat Pendek (Severely Stunted)",
            3.0: "⚠️ Pendek (Stunted)"
        }
        
        status_teks = labels_map.get(pred_idx, f"Tidak Diketahui (Index: {pred_idx})")
        
        # Saran
        saran = ""
        if "Sangat Pendek" in status_teks or "Pendek" in status_teks:
            saran = "\n\nSaran: 🩺 Pertumbuhan anak di bawah standar. Segera konsultasikan ke Posyandu/Dokter."
        elif "Normal" in status_teks:
            saran = "\n\nSaran: 👍 Pertumbuhan sehat. Pertahankan nutrisi."
        elif "Tinggi" in status_teks:
            saran = "\n\nSaran: 👌 Tumbuh kembang pesat di atas rata-rata."
            
        return status_teks + saran
        
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

# --- 4. UI GRADIO ---
ui = gr.Interface(
    fn=predict_stunting,
    inputs=[
        # Slider dibatasi max 19
        gr.Slider(minimum=0, maximum=19, step=1, label="Umur (Bulan)", info="Sesuai Dataset: 0 - 19 Bulan"),
        gr.Number(label="Tinggi Badan (cm)", value=70),
        gr.Dropdown(["Laki-laki", "Perempuan"], label="Jenis Kelamin", value="Laki-laki")
    ],
    outputs=gr.Textbox(label="Hasil Analisis"),
    title="Sistem Deteksi Dini Stunting 👶",
    description="Sistem prediksi status gizi balita menggunakan Random Forest (PySpark).",
    theme="soft"
)

if __name__ == "__main__":
    ui.launch()