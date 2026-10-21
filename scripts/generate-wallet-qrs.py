"""Regenerate checked-in wallet QR assets; not required during build or deploy.

Requires qrcode[pil]. Run from the repository root with Python 3.
"""

import json
from pathlib import Path

import qrcode
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "apps/frontend/src/lib/support.json"
OUTPUT = ROOT / "apps/frontend/public/crypto-wallets"
QR_FOREGROUND = "#09151e"
QR_BACKGROUND = "#789e94"


def main():
    wallets = json.loads(SUPPORT.read_text(encoding="utf-8"))["wallets"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = {}
    for wallet in wallets:
        filename, address = wallet["qrImage"], wallet["address"]
        if filename in generated:
            assert generated[filename] == address, "Shared QR must have the same address"
            continue
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=4)
        qr.add_data(address)
        qr.make(fit=True)
        image = qr.make_image(fill_color=QR_FOREGROUND, back_color=QR_BACKGROUND).convert("RGB")
        # Keep integer-sized modules and a quiet zone at the displayed 180px size.
        assert max(image.size) <= 180, "Increase the QR canvas for longer addresses"
        canvas = Image.new("RGB", (180, 180), QR_BACKGROUND)
        canvas.paste(image, ((180 - image.width) // 2, (180 - image.height) // 2))
        canvas.save(OUTPUT / filename, optimize=True)
        generated[filename] = address
        print(f"Generated {filename}")


if __name__ == "__main__":
    main()
