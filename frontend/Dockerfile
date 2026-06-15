# ── Stage 1: Builder ──────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# NEXT_PUBLIC_* vars must be baked in at build time — pass via --build-arg
ARG NEXT_PUBLIC_API_URL=/api/backend
ARG NEXT_PUBLIC_API_KEY=
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_KEY=$NEXT_PUBLIC_API_KEY
RUN npm run build

# ── Stage 2: Runner ──────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy toàn bộ nội dung của standalone vào thư mục hiện tại
COPY --from=builder /app/.next/standalone ./
# Copy static và public vào ĐÚNG vị trí mà standalone server yêu cầu
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

USER nextjs

EXPOSE 3000
ENV PORT 3000

# Chạy server.js từ thư mục standalone
CMD ["node", "server.js"]
