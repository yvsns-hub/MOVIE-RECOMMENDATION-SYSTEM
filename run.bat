@echo off
echo ============================================
echo   CineMatch - Movie Recommender
echo ============================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [INFO] No venv found, using system Python.
)

if not exist "data\movies_clean.csv" (
    echo [ERROR] movies_clean.csv not found.
    echo Run: python utils\data_loader.py
    pause & exit /b 1
)
if not exist "models\sim_tfidf.pkl" (
    echo [ERROR] TF-IDF model not found.
    echo Run: python utils\tfidf_engine.py
    pause & exit /b 1
)
if not exist "data\embeddings.npy" (
    echo [ERROR] BERT embeddings not found.
    echo Run: python utils\bert_engine.py
    pause & exit /b 1
)

echo [OK] All files ready.
echo Opening http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py --server.port 8501
pause
