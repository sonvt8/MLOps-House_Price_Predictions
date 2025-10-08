# House Price Prediction API

API để dự đoán giá nhà dựa trên các đặc tính khác nhau, được xây dựng với FastAPI.

## 🚀 Cách chạy API

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy API server
```bash
python run_api.py
```

API sẽ chạy tại: http://localhost:8000

### 3. Kiểm tra API documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sqft": 2000,
       "bedrooms": 3,
       "bathrooms": 2.5,
       "location": "Suburb",
       "year_built": 2010,
       "condition": "Good"
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '[
       {
         "sqft": 1500,
         "bedrooms": 2,
         "bathrooms": 1.5,
         "location": "Urban",
         "year_built": 2005,
         "condition": "Excellent"
       },
       {
         "sqft": 3000,
         "bedrooms": 4,
         "bathrooms": 3.0,
         "location": "Downtown",
         "year_built": 2015,
         "condition": "Good"
       }
     ]'
```

## 📊 API Endpoints

### GET /health
Kiểm tra trạng thái API
- **Response**: `{"status": "healthy", "model_loaded": true}`

### POST /predict
Dự đoán giá nhà cho một căn nhà
- **Request Body**:
  ```json
  {
    "sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "location": "Suburb",
    "year_built": 2010,
    "condition": "Good",
    "total_rooms": 5.5  // Optional
  }
  ```
- **Response**:
  ```json
  {
    "predicted_price": 450000.00,
    "confidence_interval": [405000.00, 495000.00],
    "features_importance": {...},
    "prediction_time": "2024-01-01T12:00:00"
  }
  ```

### POST /batch-predict
Dự đoán giá nhà cho nhiều căn nhà (tối đa 100)
- **Request Body**: Array of prediction requests
- **Response**: Array of predicted prices

## 🐛 Troubleshooting

### Model không load được
- Kiểm tra file `src/models/trained/model_pipeline.joblib` có tồn tại
- Kiểm tra file `src/models/trained/feature_names.json` có tồn tại
- Chạy training script để tạo model mới

### API không start được
- Kiểm tra port 8000 có bị chiếm không
- Kiểm tra dependencies đã cài đặt đầy đủ
- Kiểm tra Python path có đúng không

### Prediction errors
- Kiểm tra input data có đúng format không
- Kiểm tra location và condition values có hợp lệ không
- Xem logs để biết chi tiết lỗi
