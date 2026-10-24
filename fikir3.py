import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def bilgisayar_yapay_zeka_ekonomi_projesi():
    print("📊 1. ADIM: Resmi makroekonomik veriler yükleniyor...")
    
    ekonomi_veri_seti = {
        2018: {"Enflasyon": 16.33, "Faiz": 17.75, "Kur": 4.82},
        2019: {"Enflasyon": 15.18, "Faiz": 19.75, "Kur": 5.68},
        2020: {"Enflasyon": 12.28, "Faiz": 11.25, "Kur": 7.02},
        2021: {"Enflasyon": 19.60, "Faiz": 17.00, "Kur": 8.89},
        2022: {"Enflasyon": 72.31, "Faiz": 12.25, "Kur": 16.57},
        2023: {"Enflasyon": 53.86, "Faiz": 24.50, "Kur": 23.76},
        2024: {"Enflasyon": 43.57, "Faiz": 47.50, "Kur": 32.20}
    }
    
    birlesik_liste = []
    for yil in sorted(ekonomi_veri_seti.keys()):
        enflasyon = ekonomi_veri_seti[yil]["Enflasyon"]
        faiz = ekonomi_veri_seti[yil]["Faiz"]
        kur = ekonomi_veri_seti[yil]["Kur"]
        reel_faiz = round(faiz - enflasyon, 2)
        
        birlesik_liste.append({
            "Yıl": yil,
            "Enflasyon (%)": enflasyon,
            "Faiz (%)": faiz,
            "Reel Faiz (%)": reel_faiz,
            "Dolar Kuru (TL)": kur
        })

    df = pd.DataFrame(birlesik_liste)
    
    print("\n📊 BİRLEŞTİRİLMİŞ RESMİ EKONOMİ TABLOSU\n")
    print(df) # KAGGE ÖZEL 'display' YERİNE BİLGİSAYAR İÇİN 'print' YAPTIK!
    
    print("\n🤖 2. ADIM: Yapay Zeka (Doğrusal Regresyon) modeli eğitiliyor...")
    X = df[["Enflasyon (%)", "Faiz (%)", "Reel Faiz (%)"]]
    y = df["Dolar Kuru (TL)"]
    
    model = LinearRegression()
    model.fit(X, y)
    print("🎯 Model başarıyla eğitildi ve ekonomik kalıpları öğrendi!")
    
    print("\n🔮 3. ADIM: Yapay Zeka ile Gelecek Senaryosu Simülasyonu")
    farazi_senaryo = pd.DataFrame([{
        "Enflasyon (%)": 25.0,
        "Faiz (%)": 30.0,
        "Reel Faiz (%)": 5.0
    }])
    
    tahmin_kur = model.predict(farazi_senaryo)
    
    print(f"\n[Test Senaryosu Analiz Raporu]")
    print(f"-> Eğer Enflasyon %25 and Politika Faizi %30 (Pozitif Reel Faiz) olsaydı;")
    print(f"-> Matematiksel Yapay Zeka Tahmini Dolar Kuru: {round(tahmin_kur, 2)} TL")

    print("\n📈 Makroekonomik Trend Grafiği Çiziliyor (Grafik penceresini kapatana kadar program açık kalır)...")
    plt.figure(figsize=(12, 6))
    plt.plot(df["Yıl"], df["Enflasyon (%)"], marker='o', color='#d63031', linewidth=2, label="Enflasyon (TÜFE)")
    plt.plot(df["Yıl"], df["Faiz (%)"], marker='s', color='#0984e3', linewidth=2, label="Politika Faizi")
    renkler = ['#2ecc71' if x >= 0 else '#e67e22' for x in df["Reel Faiz (%)"]]
    plt.bar(df["Yıl"], df["Reel Faiz (%)"], alpha=0.25, color=renkler, label="Hesaplanan Reel Faiz")
    plt.title("Turkiye Makroekonomi Analiz Paneli")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show() # Bu komut bilgisayarınızda ayrı bir pencerede grafik açacak!

bilgisayar_yapay_zeka_ekonomi_projesi()
