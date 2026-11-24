# Foxtail Millet Fungus Detector

Detects fungus-infected areas in foxtail millet grain photos using a U-Net segmentation model (ResNet18 encoder). The app highlights infected regions in red and reports an estimated fungus area percentage.

> **Note:** The model's accuracy is modest (Dice ~0.5–0.67). Highlights are approximate estimates, not exact boundaries.

## Architecture

```
mobile/ (Expo React Native)  →  POST /analyze  →  backend/ (Node + ONNX Runtime)
                                                     ↓
                                              model/fungus_unet.onnx
```

## Prerequisites

- **Node.js** 22+ (backend)
- **Expo CLI** / Expo Go app on your phone (supports Expo SDK 57)
- Phone and laptop on the **same Wi-Fi network**

## Backend Setup

```powershell
cd backend
npm install
node server.js
# → API listening on http://0.0.0.0:3000
```

### Environment Variables

| Variable     | Default                            | Description              |
|--------------|-------------------------------------|--------------------------|
| `PORT`       | `3000`                             | Server port              |
| `MODEL_PATH` | `./model/fungus_unet.onnx`        | Path to the ONNX model   |

### Test the Backend

Place a test image (e.g., `IMG-20260110-WA0168.jpg`) in `backend/`, then:

```powershell
node test.js IMG-20260110-WA0168.jpg
# Expected: fungus area ~5.8%, result saved to result.jpg
```

### API Endpoints

| Method | Endpoint   | Description                    |
|--------|-----------|--------------------------------|
| GET    | `/health`  | `{ ok: true }`                |
| POST   | `/analyze` | Multipart `image` field → result |

**POST /analyze response:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "fungusAreaPercent": 5.8,
  "width": 960,
  "height": 1280
}
```

### Docker

```powershell
cd backend
docker build -t fungus-backend .
docker run -p 3000:3000 fungus-backend
```

### EC2 Deployment

1. Launch an instance (t3.medium+), install Node 22 and npm.
2. Clone the repo and run `npm install` in `backend/`.
3. For HTTPS, use **Caddy** as a reverse proxy:
   ```
   yourdomain.com {
     reverse_proxy localhost:3000
   }
   ```
4. For quick demos: `npx -y cloudflared tunnel --url http://localhost:3000`

## Mobile Setup

```powershell
cd mobile
npm install
npx expo start
# Scan the QR code with Expo Go on your phone
```

### Set the Server URL

1. Open the app → tap the **⚙️ gear icon** (top right)
2. Enter your laptop's LAN IP: `http://192.168.x.x:3000`
3. Tap **Test Connection** to verify
4. Tap **Save**

> **Finding your IP:** Run `ipconfig` in PowerShell, look for your Wi-Fi adapter's IPv4 address.

### Android HTTP (cleartext)

The app is already configured with `expo-build-properties` to allow cleartext HTTP traffic on Android standalone builds. For Expo Go, this works out of the box.

## Replacing the Model

1. Export your new model to ONNX with the same I/O:
   - Input: `input` — float32 `[1, 3, 256, 256]`
   - Output: `logits` — float32 `[1, 1, 256, 256]`
2. Replace `backend/model/fungus_unet.onnx`
3. Restart the server

No code changes needed. The preprocessing (ImageNet normalize, stretch to 256×256, NCHW) is fixed.

## Features

- 📷 Camera & Gallery image picker
- 🔴 Fungus area highlighting with red overlay
- 📊 Estimated fungus area percentage
- 🔀 Compare slider (before/after) + toggle views
- 💾 Save highlighted image to gallery
- 📤 Share results
- 📜 Local analysis history
- ⚙️ Configurable server URL with connection test
- ⏱ 60s request timeout with clear error messages
- 🔄 Retry on failure
