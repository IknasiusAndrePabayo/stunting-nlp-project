import gradio as gr
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import os

# 1. Inisialisasi Spark (Mode Hemat Memory)
spark = SparkSession.builder \
    .appName("StuntingDeploy") \
    .master("local[1]") \
    .config("spark.driver.memory", "512m") \
    .getOrCreate()

# 2. Load Model Folder
MODEL_PATH = "model_rf_stunting" # Pastikan nama folder sama persis

print("Sedang memuat model...")
try:
    model = PipelineModel.load(MODEL_PATH)
    print("✅ Model Random Forest berhasil dimuat!")
except Exception as e:
    model = None
    print(f"❌ Gagal memuat model: {e}")

# 3. Fungsi Prediksi
def predict_stunting(umur, tinggi, jk):
    if model is None:
        return "Error: Model tidak ditemukan."
    
    # Siapkan DataFrame input
    # (Kita isi kolom 'Status' dummy agar strukturnya cocok dengan pipeline training)
    data = pd.DataFrame({
        'Umur': [float(umur)],
        'Tinggi': [float(tinggi)],
        'JK': [jk],
        'Status': ['Normal'] 
    })
    
    spark_df = spark.createDataFrame(data)
    
    try:
        # Prediksi
        result = model.transform(spark_df)
        pred_idx = result.select("prediction").collect()[0][0]
        
        # --- MAPPING HASIL (Sesuaikan dengan output Colab Tahap 1 Tadi!) ---
        # Contoh di bawah adalah kemungkinan urutan label. 
        # GANTI teks sebelah kanan sesuai urutan print Colab Anda.
        labels = {
            0.0: "Normal",            # Biasanya data terbanyak
            1.0: "Stunting (Pendek)",
            2.0: "Sangat Pendek",
            3.0: "Tinggi"
        }
        
        status = labels.get(pred_idx, f"Tidak Diketahui (Index: {pred_idx})")
        return status
        
    except Exception as e:
        return f"Error saat prediksi: {str(e)}"

# 4. Tampilan UI Gradio
ui = gr.Interface(
    fn=predict_stunting,
    inputs=[
        gr.Number(label="Umur (bulan)", value=12),
        gr.Number(label="Tinggi Badan (cm)", value=75),
        gr.Dropdown(["Laki-laki", "Perempuan"], label="Jenis Kelamin", value="Laki-laki")
    ],
    outputs="text",
    title="Prediksi Stunting Balita (Random Forest)",
    description="Masukkan data balita untuk melihat prediksi status gizi."
)

if __name__ == "__main__":
    ui.launch()