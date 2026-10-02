import json
import os
import base64
from pathlib import Path

import requests


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE = Path("data/raw/nota-sample.png")
MODEL = os.environ["LM_STUDIO_MODEL"]

LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"


# ==========================================
# CHECK IMAGE
# ==========================================

if not IMAGE.exists():
    raise FileNotFoundError(
        f"Image tidak ditemukan: {IMAGE}"
    )


# ==========================================
# READ IMAGE
# ==========================================

image_bytes = IMAGE.read_bytes()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")


# ==========================================
# CREATE DATA URL
# ==========================================

data_url = f"data:image/png;base64,{image_base64}"


# ==========================================
# PROMPT
# ==========================================

prompt = """
Baca nota pada gambar.

Ekstrak informasi berikut:

- merchant
- tanggal
- item
- subtotal
- pajak
- total

Keluarkan HANYA JSON valid dengan struktur:

{
  "merchant": "",
  "tanggal": "",
  "item": [],
  "subtotal": 0,
  "pajak": 0,
  "total": 0
}

Aturan:
1. Jangan mengarang informasi.
2. Jika pajak tidak terlihat, isi 0.
3. Jika subtotal tidak terlihat, isi 0.
4. Jika total tidak terlihat, isi 0.
5. Untuk item, masukkan nama barang, jumlah, dan harga jika terlihat.
6. Jangan memberikan penjelasan di luar JSON.
"""


# ==========================================
# REQUEST TO LM STUDIO
# ==========================================

payload = {
    "model": MODEL,
    "input": [
        {
            "type": "text",
            "content": prompt
        },
        {
            "type": "image",
            "data_url": data_url
        }
    ],
    "temperature": 0,
    "store": False
}


response = requests.post(
    LM_STUDIO_URL,
    json=payload,
    timeout=300
)


# ==========================================
# CHECK RESPONSE
# ==========================================

response.raise_for_status()

data = response.json()


# ==========================================
# GET MODEL OUTPUT
# ==========================================

output_text = ""

for output in data.get("output", []):
    if output.get("type") == "message":
        output_text += output.get("content", "")


if not output_text:
    raise RuntimeError(
        f"LM Studio tidak mengembalikan output:\n{json.dumps(data, indent=2)}"
    )


# ==========================================
# CLEAN JSON
# ==========================================

output_text = output_text.strip()

if output_text.startswith("```json"):
    output_text = output_text[7:]

if output_text.startswith("```"):
    output_text = output_text[3:]

if output_text.endswith("```"):
    output_text = output_text[:-3]

output_text = output_text.strip()


# ==========================================
# PARSE JSON
# ==========================================

try:
    result = json.loads(output_text)
except json.JSONDecodeError:
    print("Output mentah dari model:")
    print(output_text)
    raise


# ==========================================
# SAVE RESULT
# ==========================================

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "receipt.json"

REPORT_FILE.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ==========================================
# PRINT RESULT
# ==========================================

print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)

print(f"\nHasil disimpan ke: {REPORT_FILE}")
