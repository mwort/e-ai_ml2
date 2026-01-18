import os
import cgi
import json

UPLOAD_DIR = "uploads"

def handle_file_upload(handler):
    """
    Handle incoming file upload request from an HTTP client.

    Args:
        handler: An instance of http.server.BaseHTTPRequestHandler (or subclass).
    """
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        content_type = handler.headers.get('Content-Type')
        if not content_type:
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "No Content-Type header found."}).encode("utf-8"))
            return

        ctype, pdict = cgi.parse_header(content_type)
        if ctype != 'multipart/form-data':
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Invalid content type. Expected multipart/form-data."}).encode("utf-8"))
            return

        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        pdict['CONTENT-LENGTH'] = int(handler.headers['Content-Length'])

        # Parse multipart form data
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': content_type,
            }
        )

        # Get uploaded file
        if "uploaded_file" not in form or not form["uploaded_file"].filename:
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "No file uploaded."}).encode("utf-8"))
            return

        fileitem = form["uploaded_file"]
        filename = os.path.basename(fileitem.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, 'wb') as f:
            f.write(fileitem.file.read())

        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "success",
            "message": f"✅ File '{filename}' uploaded to backend successfully."
        }).encode("utf-8"))

    except Exception as e:
        handler.send_response(500)
        handler.send_header("Content-type", "application/json")
        handler.end_headers()
        handler.wfile.write(json.dumps({
            "status": "error",
            "message": f"❌ Upload failed: {str(e)}"
        }).encode("utf-8"))
