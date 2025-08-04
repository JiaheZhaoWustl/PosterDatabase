from flask import Flask, request, jsonify, send_file
import tempfile
import os
import json
from layout_predict import predict_layout_from_data, save_grids_as_images
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    primer_path = "primer.jsonl"
    field_map = {
        "title": "title_heat",
        "location": "location_heat",
        "time": "time_heat",
        "host": "host_organization_heat",
        "purpose": "call_to_action_purpose_heat",
        "other/descriptions": "text_descriptions_details_heat"
    }
    selected_fields = [field_map.get(f, f) for f in data.get("selectedOptions", [])]
    user_data = json.dumps(data)
    grids = predict_layout_from_data(
        user_data=user_data,
        primer_path=primer_path,
        selected_fields=selected_fields
    )
    frame_w = data["frame"]["width"]
    frame_h = data["frame"]["height"]
    with tempfile.TemporaryDirectory() as tmpdir:
        filepaths = save_grids_as_images(grids, tmpdir, frame_w, frame_h)
        # For demo: return all images as base64
        import base64
        results = {}
        for field, path in filepaths.items():
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                results[field] = b64
        return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)