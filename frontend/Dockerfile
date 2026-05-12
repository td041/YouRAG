FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (sử dụng npm ci để cài đặt chính xác các versions)
RUN npm ci

# Copy toàn bộ mã nguồn frontend
COPY . .

# Build Next.js
RUN npm run build

# Runner stage
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV production

# Copy thư mục build và node_modules từ stage trước
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

ENV PORT 3000

CMD ["npm", "start"]
