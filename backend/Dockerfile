FROM node:22-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .

ENV PORT=3000
ENV MODEL_PATH=./model/millet_multitask_unet.onnx

EXPOSE 3000

# onnxruntime-node needs to download binaries; ensure they're cached in the image
RUN node -e "require('onnxruntime-node')"

CMD ["node", "server.js"]
