from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import datetime
import sys

database_url = 'sqlite:///./predictions.db'
engine = create_engine(database_url, connect_args={"check_same_thread": False})
session_local = sessionmaker(autocommit = False, autoflush = False, bind = engine)
base = declarative_base()
class prediction_record(base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=datetime.datetime.utcnow)
    brand = Column(String, index = True)
    processor_name = Column(String)
    ram_gb = Column(Integer)
    ssd_gb = Column(Integer)
    pred_floor = Column(Float)
    pred_median = Column(Float)
    pred_ceiling = Column(Float)

base.metadata.create_all(bind = engine)

app = FastAPI(title="Laptop Price Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()

class feature_engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, x, y = None):
        return self
    def transform(self, x):
        h = x.copy()
        if 'hdd' in h.columns:
            h['is_ssd'] = h['hdd'] == '0 GB'
        else:
            h['is_ssd'] = False
        if 'processor_gnrtn' in h.columns and 'graphic_card_gb' in h.columns:
            is_anomaly_high_spec = (
                (h['processor_gnrtn'] == 'Not Available') & 
                (h['is_ssd'] == True) & 
                (h['graphic_card_gb'] != '0 GB')
            )
            h['processor_gnrtn'] = np.where(is_anomaly_high_spec, '12th', h['processor_gnrtn'])
            h['processor_gnrtn'] = np.where(
                h['processor_gnrtn'] == 'Not Available', 
                'other', 
                h['processor_gnrtn']
            )
        if 'graphic_card_gb' in h.columns:
            h['graphic_card_gb'] = h['graphic_card_gb'].astype(str).apply(lambda x: int(x.split()[0] if len(x.split()) > 0 else 0))
        if 'ram_gb' in h.columns:
            h['ram_gb'] = h['ram_gb'].astype(str).apply(lambda x: int(x.split()[0] if len(x.split()) > 0 else 0))
        if 'ssd' in h.columns:
            h['ssd'] = h['ssd'].astype(str).apply(lambda x: int(x.split()[0] if len(x.split()) > 0 else 0))
        if 'rating' in h.columns:
            h['rating'] = h['rating'].astype(str).apply(lambda x: int(x.split()[0] if len(x.split()) > 0 else 0))
        return h
sys.modules['__main__'].feature_engineering = feature_engineering
try:
    model_low = joblib.load('model_low.joblib')
    model_medium = joblib.load('model_med.joblib')
    model_high = joblib.load('model_high.joblib')
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

class laptop_specs(BaseModel):
    brand: str = Field(..., example="ASUS")
    processor_brand: str = Field(..., example="Intel")
    processor_name: str = Field(..., example="Core i7")
    processor_gnrtn: str = Field(..., example="12th")
    ram_gb: str = Field(..., example="16 GB")
    ram_type: str = Field(..., example="DDR4")
    ssd: str = Field(..., example="512 GB")
    hdd: str = Field(..., example="0 GB")
    os: str = Field(..., example="Windows 11")
    os_bit: str = Field(..., example="64-bit")
    graphic_card_gb: str = Field(..., example="4 GB")
    weight: str = Field(..., example="Casual Laptop")
    warranty: str = Field(..., example="1 year")
    Touchscreen: str = Field(..., example="No")
    msoffice: str = Field(..., example="No")
    rating: str = Field(..., example="4 stars")
    Number_of_Ratings: float = Field(..., alias="Number of Ratings", example=150.0)
    Number_of_Reviews: float = Field(..., alias="Number of Reviews", example=30.0)
    class config:
        populate_by_name = True

@app.post("/predict", status_code=200)
def predict_price(laptop: laptop_specs, db: Session = Depends(get_db)):
    try:
        raw_data = laptop.model_dump(by_alias=True)
        input = pd.DataFrame([raw_data])
        floor = float(model_low.predict(input)[0])
        median = float(model_medium.predict(input)[0])
        ceiling = float(model_high.predict(input)[0])
        if not(floor <= median <= ceiling):
            floor, median, ceiling = sorted([floor, median, ceiling])
        
        db_record = prediction_record(
            brand = laptop.brand,
            processor_name = laptop.processor_name,
            ram_gb = int(laptop.ram_gb.split()[0]) if len(laptop.ram_gb.split()) > 0 else 0,
            ssd_gb = int(laptop.ssd.split()[0]) if len(laptop.ssd.split()) > 0 else 0,
            pred_floor = round(floor, 2),
            pred_median = round(median, 2),
            pred_ceiling = round(ceiling, 2)
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        return {
            "status": "success",
            "transaction_metadata": {
                "saved_log_id": db_record.id,
                "timestamp_utc": db_record.timestamp
            },
            "pricing_appraisal": {
                "bargain_floor_15th": db_record.predicted_floor,
                "fair_market_median_50th": db_record.predicted_median,
                "premium_ceiling_85th": db_record.predicted_ceiling
            }
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Production Engine Core Error. Transaction safely rolled back. Trace: {str(e)}"
        )
