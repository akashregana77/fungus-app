// Usage: node test.js path/to/image.jpg [http://localhost:3000/analyze]
// Saves the highlighted image as result.jpg and prints the fungus area %.
const fs = require('fs');

const imgPath = process.argv[2];
const url = process.argv[3] || 'http://localhost:3000/analyze';

if (!imgPath) {
  console.error('Usage: node test.js path/to/image.jpg [url]');
  process.exit(1);
}

(async () => {
  const form = new FormData();
  form.append('image', new Blob([fs.readFileSync(imgPath)], { type: 'image/jpeg' }), 'sample.jpg');

  const t0 = Date.now();
  const res = await fetch(url, { method: 'POST', body: form });
  const json = await res.json();

  console.log('status:', res.status, '| time:', Date.now() - t0, 'ms');
  if (json.error) return console.error('error:', json.error);

  console.log('isMillet:', json.isMillet, '| confidence:', json.confidence, '| maxLogit:', json.maxLogit, '| texture:', json.texture);
  if (!json.isMillet) {
    console.log('Rejected:', json.message);
    return;
  }

  console.log('millet:', json.millet, '| fungus area %:', json.fungusAreaPercent, '| size:', json.width, 'x', json.height);
  fs.writeFileSync('result.jpg', Buffer.from(json.image.split(',')[1], 'base64'));
  console.log('saved result.jpg');
})();
