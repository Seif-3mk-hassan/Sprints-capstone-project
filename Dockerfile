# ==========================================
# Stage 1: Builder (ETL Stage)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# 1. نسخ ملف المتطلبات وتنزيل المكتبات
COPY requirements.txt . 
RUN pip install --user --no-cache-dir -r requirements.txt 

# 2. نسخ الملفات المطلوبة لعملية الـ ETL فقط
COPY etl_pipeline.py reviews.csv ./ 

# 3. تشغيل الـ ETL لبناء قاعدة البيانات من ملف الـ CSV
# السطر ده هيقرأ reviews.csv ويطلع reviews_db.sqlite
RUN python etl_pipeline.py 

# ==========================================
# Stage 2: Runner (API Stage - Final Image)
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# 1. نقل المكتبات اللي اتسطبت من المرحلة الأولى
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 2. نسخ كود التطبيق الأساسي فقط
COPY app.py models.py ./ 

# 3. نسخ قاعدة البيانات الجاهزة فقط من مرحلة الـ Builder
# كده إحنا سيبنا ملف الـ reviews.csv التقيل ورا ومخدناش غير الخلاصة
COPY --from=builder /app/reviews_db.sqlite . 

# 4. فتح البورت وتشغيل السيرفر
EXPOSE 8000

# تشغيل الـ FastAPI باستخدام uvicorn (مباشرة من app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]