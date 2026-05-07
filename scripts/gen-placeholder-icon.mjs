// 1024x1024 のプレースホルダ PNG を生成する。
// 依存ゼロ (Node 標準の zlib のみ) で動かすため、PNG をスクラッチで構築する。
//
// 使い方:
//   node scripts/gen-placeholder-icon.mjs <出力パス>
import { writeFileSync } from "node:fs";
import { deflateSync } from "node:zlib";

const SIZE = 1024;

function crc32(buf) {
  let c;
  const table = [];
  for (let n = 0; n < 256; n++) {
    c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, "ascii");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crc]);
}

function makePng(width, height, pixels) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    const rowStart = y * (width * 4 + 1);
    raw[rowStart] = 0; // filter
    pixels.copy(raw, rowStart + 1, y * width * 4, (y + 1) * width * 4);
  }
  const compressed = deflateSync(raw);

  return Buffer.concat([
    sig,
    chunk("IHDR", ihdr),
    chunk("IDAT", compressed),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const out = process.argv[2] ?? "src-tauri/icons/app-icon.png";

const pixels = Buffer.alloc(SIZE * SIZE * 4);
const cx = SIZE / 2;
const cy = SIZE / 2;
const cornerRadius = 180;

function rgba(r, g, b, a = 255) {
  return [r, g, b, a];
}

function setPixel(x, y, [r, g, b, a]) {
  const i = (y * SIZE + x) * 4;
  pixels[i] = r;
  pixels[i + 1] = g;
  pixels[i + 2] = b;
  pixels[i + 3] = a;
}

function isInsideRoundedRect(x, y, left, top, right, bottom, radius) {
  if (x < left || x > right || y < top || y > bottom) return false;
  const dx = Math.max(left + radius - x, 0, x - (right - radius));
  const dy = Math.max(top + radius - y, 0, y - (bottom - radius));
  return dx * dx + dy * dy <= radius * radius;
}

const bgTopColor = [15, 61, 46];
const bgBottomColor = [31, 111, 74];
const cardColor = [250, 250, 245];
const cardBorder = [29, 58, 42];
const accent = [29, 58, 42];

for (let y = 0; y < SIZE; y++) {
  const t = y / (SIZE - 1);
  const r = Math.round(bgTopColor[0] + (bgBottomColor[0] - bgTopColor[0]) * t);
  const g = Math.round(bgTopColor[1] + (bgBottomColor[1] - bgTopColor[1]) * t);
  const b = Math.round(bgTopColor[2] + (bgBottomColor[2] - bgTopColor[2]) * t);
  for (let x = 0; x < SIZE; x++) {
    if (isInsideRoundedRect(x, y, 0, 0, SIZE - 1, SIZE - 1, cornerRadius)) {
      setPixel(x, y, rgba(r, g, b));
    } else {
      setPixel(x, y, rgba(0, 0, 0, 0));
    }
  }
}

const cardLeft = 232;
const cardTop = 172;
const cardRight = 791;
const cardBottom = 851;
const cardRadius = 64;
const borderWidth = 14;

for (let y = cardTop - borderWidth; y <= cardBottom + borderWidth; y++) {
  for (let x = cardLeft - borderWidth; x <= cardRight + borderWidth; x++) {
    const inOuter = isInsideRoundedRect(
      x,
      y,
      cardLeft - borderWidth,
      cardTop - borderWidth,
      cardRight + borderWidth,
      cardBottom + borderWidth,
      cardRadius + borderWidth,
    );
    const inInner = isInsideRoundedRect(
      x,
      y,
      cardLeft,
      cardTop,
      cardRight,
      cardBottom,
      cardRadius,
    );
    if (inInner) {
      setPixel(x, y, rgba(...cardColor));
    } else if (inOuter) {
      setPixel(x, y, rgba(...cardBorder));
    }
  }
}

// Draw a stylized "雀" tile front: a smaller dark green rounded rect
// representing a mahjong tile back, with three accent dots.
const tileLeft = 332;
const tileTop = 282;
const tileRight = 692;
const tileBottom = 702;
const tileRadius = 40;
const tileColor = [21, 86, 58];

for (let y = tileTop; y <= tileBottom; y++) {
  for (let x = tileLeft; x <= tileRight; x++) {
    if (
      isInsideRoundedRect(
        x,
        y,
        tileLeft,
        tileTop,
        tileRight,
        tileBottom,
        tileRadius,
      )
    ) {
      setPixel(x, y, rgba(...tileColor));
    }
  }
}

function drawDot(cx, cy, radius, color) {
  for (let y = cy - radius; y <= cy + radius; y++) {
    for (let x = cx - radius; x <= cx + radius; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= radius * radius) {
        setPixel(x, y, rgba(...color));
      }
    }
  }
}

const dotColor = [232, 236, 222];
const dotRadius = 38;
drawDot(420, 412, dotRadius, dotColor);
drawDot(512, 492, dotRadius, dotColor);
drawDot(604, 572, dotRadius, dotColor);

const png = makePng(SIZE, SIZE, pixels);
writeFileSync(out, png);
console.log(`Wrote ${out} (${png.length} bytes)`);
