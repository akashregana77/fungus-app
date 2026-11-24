// Bilinear resize on interleaved (HWC) data, matching OpenCV's cv2.resize(..., INTER_LINEAR):
// half-pixel centers, edge clamping, no antialiasing. Returns a Float32Array.
function resizeBilinear(src, sw, sh, dw, dh, ch) {
  if (src.length !== sw * sh * ch) {
    throw new Error(`resizeBilinear: expected ${sw * sh * ch} values, got ${src.length}`);
  }
  const dst = new Float32Array(dw * dh * ch);
  const scaleX = sw / dw;
  const scaleY = sh / dh;

  const x0s = new Int32Array(dw);
  const x1s = new Int32Array(dw);
  const wxs = new Float32Array(dw);
  for (let x = 0; x < dw; x++) {
    let fx = (x + 0.5) * scaleX - 0.5;
    if (fx < 0) fx = 0;
    let x0 = Math.floor(fx);
    if (x0 > sw - 1) x0 = sw - 1;
    x0s[x] = x0;
    x1s[x] = Math.min(x0 + 1, sw - 1);
    wxs[x] = fx - x0;
  }

  for (let y = 0; y < dh; y++) {
    let fy = (y + 0.5) * scaleY - 0.5;
    if (fy < 0) fy = 0;
    let y0 = Math.floor(fy);
    if (y0 > sh - 1) y0 = sh - 1;
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = fy - y0;
    const row0 = y0 * sw;
    const row1 = y1 * sw;

    for (let x = 0; x < dw; x++) {
      const wx = wxs[x];
      const i00 = (row0 + x0s[x]) * ch;
      const i01 = (row0 + x1s[x]) * ch;
      const i10 = (row1 + x0s[x]) * ch;
      const i11 = (row1 + x1s[x]) * ch;
      const o = (y * dw + x) * ch;
      for (let c = 0; c < ch; c++) {
        const top = src[i00 + c] * (1 - wx) + src[i01 + c] * wx;
        const bottom = src[i10 + c] * (1 - wx) + src[i11 + c] * wx;
        dst[o + c] = top * (1 - wy) + bottom * wy;
      }
    }
  }
  return dst;
}

module.exports = { resizeBilinear };
