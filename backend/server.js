const fs = require('fs');
const path = require('path');
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const sharp = require('sharp');
const ort = require('onnxruntime-node');
const { resizeBilinear } = require('./resize');

const PORT = process.env.PORT || 3000;
const MODEL_PATH = process.env.MODEL_PATH || path.join(__dirname, 'model', 'millet_multitask_unet.onnx');
const LABELS_PATH = process.env.LABELS_PATH || path.join(__dirname, 'model', 'labels.json');

// Below this millet confidence the app should warn the user (override with LOW_CONFIDENCE=0.8 etc.)
const LOW_CONFIDENCE = parseFloat(process.env.LOW_CONFIDENCE || '0.7');

// Thresholds to reject photos that are not millet
// Real millet grain photos have high texture (~15 to 30), whereas faces, skin, rooms, walls are < 5.
const MIN_CONFIDENCE = parseFloat(process.env.MIN_CONFIDENCE || '0.90');
const MIN_LOGIT      = parseFloat(process.env.MIN_LOGIT || '0');
const MIN_TEXTURE    = parseFloat(process.env.MIN_TEXTURE || '9.0');

// Millets whose fungus mask was trained on enough annotated images to trust the percentage.
// Update this list after round 2 (more masks for kodo/little/pearl).
const SEG_RELIABLE = new Set((process.env.SEG_RELIABLE || 'foxtail').split(',').map((s) => s.trim()));

const MAX_SIDE = 1280; // cap the size of the image we process and return

// Loaded from labels.json at startup so the server always matches the trained model
let MILLETS;
let SIZE = 256;
let MEAN = [0.485, 0.456, 0.406];
let STD = [0.229, 0.224, 0.225];

let session;

function softmax(arr) {
  let max = -Infinity;
  for (const v of arr) if (v > max) max = v;
  const exps = Array.from(arr, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function analyze(buffer) {
  // 1. Decode, apply EXIF rotation (phone photos), cap size, get raw 8-bit RGB
  const { data: rgb, info } = await sharp(buffer)
    .rotate()
    .resize({ width: MAX_SIDE, height: MAX_SIDE, fit: 'inside', withoutEnlargement: true })
    .removeAlpha()
    .toColourspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true });
  const { width: w, height: h, channels } = info;
  if (channels !== 3 || rgb.length !== w * h * 3) {
    throw new Error(`Unexpected pixel layout: ${channels} channels, ${rgb.length} bytes for ${w}x${h}`);
  }

  // 2. Model input: stretch to SIZE x SIZE (rounded to 8-bit like cv2.resize on a uint8 image)
  const small = resizeBilinear(rgb, w, h, SIZE, SIZE, 3);

  // Texture score: convert small (uint8 RGB 256x256) to gray g = 0.299R + 0.587G + 0.114B (float).
  // tex = ( mean(|g[y][x+1]-g[y][x]|) + mean(|g[y+1][x]-g[y][x]|) ) / 2
  const gray = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const rowOffset = y * SIZE;
    for (let x = 0; x < SIZE; x++) {
      const idx = (rowOffset + x) * 3;
      const r = Math.round(small[idx]);
      const g = Math.round(small[idx + 1]);
      const b = Math.round(small[idx + 2]);
      gray[rowOffset + x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  let sumDiffH = 0;
  for (let y = 0; y < SIZE; y++) {
    const rowOffset = y * SIZE;
    for (let x = 0; x < SIZE - 1; x++) {
      sumDiffH += Math.abs(gray[rowOffset + x + 1] - gray[rowOffset + x]);
    }
  }
  const meanDiffH = sumDiffH / (SIZE * (SIZE - 1));

  let sumDiffV = 0;
  for (let y = 0; y < SIZE - 1; y++) {
    const rowOffset = y * SIZE;
    const nextRowOffset = (y + 1) * SIZE;
    for (let x = 0; x < SIZE; x++) {
      sumDiffV += Math.abs(gray[nextRowOffset + x] - gray[rowOffset + x]);
    }
  }
  const meanDiffV = sumDiffV / ((SIZE - 1) * SIZE);

  const tex = (meanDiffH + meanDiffV) / 2;

  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (Math.round(small[i * 3 + c]) / 255 - MEAN[c]) / STD[c];
    }
  }

  // 3. Inference: the multi-task model has TWO outputs (mask + millet type)
  const out = await session.run({ input: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const maskLogits = out.mask_logits.data;
  const classLogits = out.class_logits.data;

  // 4a. Millet type: softmax over class logits
  const probs = softmax(classLogits);
  let best = 0;
  let maxLogit = -Infinity;
  for (let i = 0; i < classLogits.length; i++) {
    if (classLogits[i] > maxLogit) maxLogit = classLogits[i];
    if (probs[i] > probs[best]) best = i;
  }
  const millet = MILLETS[best];
  const confidence = probs[best];

  const minConfidence = process.env.MIN_CONFIDENCE !== undefined ? parseFloat(process.env.MIN_CONFIDENCE) : MIN_CONFIDENCE;
  const minLogit = process.env.MIN_LOGIT !== undefined ? parseFloat(process.env.MIN_LOGIT) : MIN_LOGIT;
  const minTexture = process.env.MIN_TEXTURE !== undefined ? parseFloat(process.env.MIN_TEXTURE) : MIN_TEXTURE;

  const isMillet = confidence >= minConfidence && maxLogit >= minLogit && tex >= minTexture;

  const probabilities = {};
  MILLETS.forEach((m, i) => {
    probabilities[m] = Math.round(probs[i] * 1000) / 1000;
  });

  const roundedConfidence = Math.round(confidence * 1000) / 1000;
  const roundedMaxLogit = Math.round(maxLogit * 1000) / 1000;
  const roundedTexture = Math.round(tex * 1000) / 1000;

  // If isMillet is false, return early and skip the mask/overlay work
  if (!isMillet) {
    return {
      isMillet: false,
      millet: null,
      confidence: roundedConfidence,
      probabilities,
      maxLogit: roundedMaxLogit,
      texture: roundedTexture,
      message: "This does not look like millet grains. Please upload a clear close-up photo of foxtail, kodo, little or pearl millet.",
    };
  }

  // 4b. Fungus mask: sigmoid, then bilinear upscale of the probability map to the image size
  const prob = new Float32Array(plane);
  for (let i = 0; i < plane; i++) prob[i] = 1 / (1 + Math.exp(-maskLogits[i]));
  const probFull = resizeBilinear(prob, SIZE, SIZE, w, h, 1);

  // 5. Threshold at 0.5 and blend red over fungus pixels
  const outRgb = Buffer.from(rgb);
  let fungusPixels = 0;
  for (let i = 0; i < w * h; i++) {
    if (probFull[i] > 0.5) {
      fungusPixels++;
      const o = i * 3;
      outRgb[o] = Math.round(rgb[o] * 0.6 + 255 * 0.4);
      outRgb[o + 1] = Math.round(rgb[o + 1] * 0.6);
      outRgb[o + 2] = Math.round(rgb[o + 2] * 0.6);
    }
  }

  const jpeg = await sharp(outRgb, { raw: { width: w, height: h, channels: 3 } })
    .jpeg({ quality: 85 })
    .toBuffer();

  return {
    isMillet: true,
    maxLogit: roundedMaxLogit,
    texture: roundedTexture,
    image: `data:image/jpeg;base64,${jpeg.toString('base64')}`,
    fungusAreaPercent: Math.round((1000 * fungusPixels) / (w * h)) / 10,
    width: w,
    height: h,
    // new fields (the existing client keeps working, it just ignores these)
    millet,
    confidence: roundedConfidence,
    probabilities,
    lowConfidence: confidence < LOW_CONFIDENCE,
    fungusReliable: SEG_RELIABLE.has(millet),
  };
}

async function main() {
  const labels = JSON.parse(fs.readFileSync(LABELS_PATH, 'utf8'));
  MILLETS = labels.millets;
  SIZE = labels.size || SIZE;
  MEAN = labels.mean || MEAN;
  STD = labels.std || STD;
  console.log('Millets:', MILLETS.join(', '), '| input size:', SIZE);

  session = await ort.InferenceSession.create(MODEL_PATH);
  console.log('Model loaded:', MODEL_PATH);
  console.log('Model outputs:', session.outputNames.join(', '));
  for (const name of ['mask_logits', 'class_logits']) {
    if (!session.outputNames.includes(name)) {
      throw new Error(`Model is missing output "${name}". Is this the multi-task model?`);
    }
  }

  // warm-up so the first real request isn't slow
  await session.run({ input: new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });

  const app = express();
  app.use(cors());
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));
  const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

  app.get('/health', (_req, res) => res.json({ ok: true, millets: MILLETS }));

  app.post('/analyze', upload.single('image'), async (req, res) => {
    let buffer;
    if (req.file && req.file.buffer && req.file.buffer.length > 0) {
      const headerStr = req.file.buffer.slice(0, 15).toString();
      if (headerStr.startsWith('[object Object]')) {
        console.warn('Received [object Object] in multipart field "image" from React Native');
        return res.status(400).json({
          error: 'Client sent invalid image data ([object Object]). Update mobile client to use base64 upload.',
        });
      }
      buffer = req.file.buffer;
    } else if (req.body && (req.body.image || req.body.base64)) {
      const str = req.body.image || req.body.base64;
      const b64 = str.replace(/^data:image\/[a-zA-Z0-9+.-]+;base64,/, '');
      buffer = Buffer.from(b64, 'base64');
    }

    if (!buffer || buffer.length === 0) {
      return res.status(400).json({
        error: 'Send an image in the "image" field or as a JSON { image: "<base64>" } payload.',
      });
    }

    try {
      const t0 = Date.now();
      const result = await analyze(buffer);
      console.log(
        `analyzed image (${buffer.length} bytes) in ${Date.now() - t0} ms -> isMillet: ${result.isMillet}, millet: ${result.millet} ` +
          `(${(result.confidence * 100).toFixed(1)}%, maxLogit: ${result.maxLogit}, tex: ${result.texture})${result.isMillet ? `, fungus ${result.fungusAreaPercent}%` : ''}`
      );
      res.json(result);
    } catch (err) {
      console.error('Analysis error:', err);
      res.status(500).json({ error: err.message || 'Could not process this image.' });
    }
  });

  const server = app.listen(PORT, '0.0.0.0', () => console.log(`API listening on http://0.0.0.0:${PORT}`));
  server.timeout = 120_000; // 120s request timeout
}

main().catch((err) => {
  console.error('Failed to start:', err);
  process.exit(1);
});