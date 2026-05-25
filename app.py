from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

# Uygulama başlatma
app = FastAPI(title="Smart-LogiX Tahmin Servisi")

# CORS Desteği Ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modeli yükle - mutlak yol ile çalışmasını sağla
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "logistics_delay_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı! Beklenen yol: {model_path}")

model_data = joblib.load(model_path)
model = model_data['model']
feature_names = model_data['features']

print(f"✅ Model başarıyla yüklendi: {model_path}")
print(f"Beklenen özellikler: {feature_names}")

# .NET'ten gelecek verinin şeması (Validation)
class LogisticsInput(BaseModel):
    Asset_ID: int
    Shipment_Status: int
    Traffic_Status: int
    Logistics_Delay_Reason: int
    Month: int
    Day_of_Week: int
    Hour: int

@app.get("/")
async def root():
    return {
        "message": "Smart-LogiX Tahmin Servisi çalışıyor!",
        "endpoints": {
            "predict": {
                "method": "POST",
                "url": "http://localhost:8000/predict",
                "description": "Gecikme tahminleme"
            },
            "health": {
                "method": "GET",
                "url": "http://localhost:8000/health",
                "description": "Sistem sağlık kontrolü"
            },
            "docs": {
                "method": "GET",
                "url": "http://localhost:8000/docs",
                "description": "Interactive API belgeleri (Swagger)"
            }
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "features": feature_names
    }

@app.post("/predict")
async def predict(data: LogisticsInput):
    try:
        # Gelen veriyi DataFrame yapısına çevir
        df = pd.DataFrame([data.dict()])
        
        # Sütunları modelin eğitimdeki sırasına göre hizala
        df = df[feature_names]
        
        # Tahmin yap
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)
        confidence = float(np.max(probabilities))
        
        return {
            "is_delayed": int(prediction[0]),
            "confidence": confidence,
            "probability_no_delay": float(probabilities[0][0]),
            "probability_delayed": float(probabilities[0][1])
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Eksik alan: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hata: {str(e)}")