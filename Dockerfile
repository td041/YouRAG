FROM python:3.12-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết (nếu có)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Tắt virtualenvs của Poetry vì Docker container bản thân nó đã bị cô lập
RUN poetry config virtualenvs.create false

# Copy các file quản lý dependencies
COPY pyproject.toml poetry.lock ./

# Cài đặt các dependencies (chỉ cài đặt main dependencies, bỏ qua dev)
RUN poetry install --only main --no-root

# Copy toàn bộ mã nguồn
COPY . .

# Expose cổng 8000 cho FastAPI
EXPOSE 8000

# Khởi chạy server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
