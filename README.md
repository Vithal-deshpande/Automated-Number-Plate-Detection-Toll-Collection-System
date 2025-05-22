from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import pytesseract
from datetime import datetime

app = Flask(__name__)

# In-memory toll account database
tollDB = {
    "KA32EQ5990": {"owner": "VITHAL DESHPANDE", "balance": 50.00},
    "MH12NESS922": {"owner": "Jane Smith", "balance": 15.00},
    "PGMN112": {"owner": "Company Fleet", "balance": 300.00},
    "MNO2345": {"owner": "Alice Johnson", "balance": 7.00},
}

TOLL_FEE = 5.00
transaction_log = []

# Simple plate detection function using OpenCV + pytesseract OCR
def detect_number_plate(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(filtered, 30, 200)

    # Find contours and sort by size descending
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for cnt in contours:
        # Approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        # If contour has 4 vertices, it could be a plate
        if len(approx) == 4:
            plate_contour = approx
            break
    if plate_contour is None:
        return None  # Plate area not found

    # Mask everything except plate contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Crop the plate area from the image using bounding rect
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = gray[y:y+h, x:x+w]

    # OCR on plate image
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(plate_img, config=custom_config)
    # Clean text - remove non-alphanumeric, upper case
    plate_text = ''.join(filter(str.isalnum, text)).upper()

    if not plate_text:
        return None
    return plate_text

def add_transaction_log(entry):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    transaction_log.insert(0, f"[{timestamp}] {entry}")
    # Limit log size
    if len(transaction_log) > 100:
        transaction_log.pop()

@app.route('/')
def index():
    # Serve single HTML file frontend with upload form and JS
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Number Plate Detection & Automated Toll Collection</title>
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #1c2833, #2980b9);
    margin: 0; padding: 0; min-height: 100vh;
    color: #eee;
    display: flex; flex-direction: column; align-items: center;
}
header {
    font-size: 2.4rem; font-weight: 700; margin: 2rem 0 1rem;
    text-shadow: 0 0 10px rgba(0,0,0,0.5);
}
main {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 2rem;
    max-width: 520px;
    width: 90%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
input[type=file] {
    padding: 0.3rem;
    border-radius: 6px;
    border: none;
    outline: none;
    font-size: 1rem;
}
button {
    background: #27ae60;
    color: #fff;
    border: none;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.75rem;
    border-radius: 8px;
    cursor: pointer;
    text-shadow: 0 0 6px rgba(0,0,0,0.3);
    transition: background-color 0.3s ease;
}
button:hover {
    background: #2ecc71;
}
#output {
    margin-top: 1.5rem;
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    padding: 1rem;
    min-height: 3rem;
    font-size: 1.2rem;
    font-weight: 600;
    word-wrap: break-word;
}
#log {
    margin-top: 2rem;
    max-height: 200px;
    overflow-y: auto;
    background: rgba(0,0,0,0.25);
    border-radius: 12px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.87rem;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.4);
    color: #7fff7f;
}
#log h2 {
    margin-top: 0;
}
.log-entry {
    border-bottom: 1px solid rgba(127,255,127,0.3);
    padding: 0.2rem 0;
}
</style>
</head>
<body>
<header>🚘 Number Plate Detection & Automated Toll Collection</header>
<main>
    <form id="uploadForm" aria-label="Upload vehicle image for detection">
        <label for="imageInput">Upload vehicle image:</label>
        <input type="file" accept="image/*" id="imageInput" required />
        <button type="submit">Detect Number Plate & Charge Toll</button>
    </form>
    <div id="output" aria-live="polite"></div>
    <section id="log" aria-live="polite" aria-label="Transaction Log">
        <h2>Transaction Log</h2>
        <div id="logEntries"></div>
    </section>
</main>
<script>
const form = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const output = document.getElementById('output');
const logEntries = document.getElementById('logEntries');

async function fetchTransactionLog() {
    try {
        const res = await fetch('/transactions');
        if (!res.ok) throw new Error('Failed to fetch logs');
        const logs = await res.json();
        logEntries.innerHTML = '';
        logs.forEach(entry => {
            const div = document.createElement('div');
            div.className = 'log-entry';
            div.textContent = entry;
            logEntries.appendChild(div);
        });
    } catch(e) {
        logEntries.textContent = 'Error loading logs.';
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    output.textContent = 'Processing image, please wait...';
    const file = imageInput.files[0];
    if (!file) {
        output.textContent = 'Please select an image file.';
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        // Send image for detection
        const detectRes = await fetch('/detect_plate', {
            method: 'POST',
            body: formData
        });

        if (!detectRes.ok) {
            const err = await detectRes.json();
            output.textContent = 'Detection error: ' + (err.error || detectRes.statusText);
            return;
        }

        const detectData = await detectRes.json();
        const plate = detectData.plate;
        if (!plate) {
            output.textContent = 'Unable to detect number plate. Please try a clearer image.';
            return;
        }
        output.textContent = `Detected Number Plate: ${plate} - Charging Toll...`;

        // Charge toll
        const chargeRes = await fetch('/charge_toll', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ plate })
        });

        if (!chargeRes.ok) {
            const err = await chargeRes.json();
            output.textContent = `Toll charging failed: ${err.error || chargeRes.statusText}`;
            await fetchTransactionLog();
            return;
        }

        const chargeData = await chargeRes.json();
        output.textContent = `Toll charged: $${chargeData.feeCharged.toFixed(2)}\nVehicle: ${chargeData.plate} (${chargeData.owner})\nRemaining balance: $${chargeData.newBalance.toFixed(2)}`;
        await fetchTransactionLog();
    } catch (err) {
        output.textContent = 'Server error or network issue: ' + err.message;
    }
});

window.onload = () => {
    fetchTransactionLog();
};
</script>
</body>
</html>
    ''')

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({"error": "No image file included."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Read image bytes and convert to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image file."}), 400

    plate_text = detect_number_plate(img)
    if not plate_text:
        return jsonify({"plate": None})

    return jsonify({"plate": plate_text})


@app.route('/charge_toll', methods=['POST'])
def charge_toll():
    data = request.get_json()
    if not data or 'plate' not in data:
        return jsonify({"error": "License plate is required."}), 400
    plate = data['plate'].upper()
    if plate not in tollDB:
        add_transaction_log(f"Unrecognized vehicle {plate} attempted toll charge.")
        return jsonify({"error": "Vehicle not recognized in toll database."}), 404

    driver = tollDB[plate]
    if driver['balance'] < TOLL_FEE:
        add_transaction_log(f"Low balance for vehicle {plate}. Toll NOT charged.")
        return jsonify({"error": "Insufficient balance, please recharge account."}), 403
    
    driver['balance'] -= TOLL_FEE
    add_transaction_log(f"Charged ${TOLL_FEE:.2f} to {plate} (Owner: {driver['owner']}). New balance: ${driver['balance']:.2f}")

    return jsonify({
        "message": "Toll charged successfully.",
        "plate": plate,
        "owner": driver['owner'],
        "newBalance": driver['balance'],
        "feeCharged": TOLL_FEE,
    })

@app.route('/transactions', methods=['GET'])
def transactions():
    return jsonify(transaction_log)


if __name__ == "__main__":
    app.run(debug=True)

