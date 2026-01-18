<?php
error_reporting(E_ALL);
ini_set("display_errors", 1);
header("Content-Type: application/json");


file_put_contents("upload_debug.log", print_r($_FILES, true));

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["error" => "Invalid request method. Use POST."]);
    exit;
}

if (!isset($_FILES['uploaded_file']) || $_FILES['uploaded_file']['error'] !== UPLOAD_ERR_OK) {
    echo json_encode(["error" => "No valid file uploaded."]);
    exit;
}

$uploadDir = __DIR__ . "/uploads";
if (!is_dir($uploadDir)) {
    if (!mkdir($uploadDir, 0775, true)) {
        echo json_encode(["error" => "Failed to create upload directory."]);
        exit;
    }
}

$tmpName = $_FILES['uploaded_file']['tmp_name'];
$originalName = basename($_FILES['uploaded_file']['name']);
$targetPath = $uploadDir . "/" . $originalName;

if (!move_uploaded_file($tmpName, $targetPath)) {
    echo json_encode(["error" => "Failed to move uploaded file."]);
    exit;
}

// ✅ FORWARD to Python backend
$backendUrl = "http://localhost:8001/upload";

$curl = curl_init();
$cfile = new CURLFile($targetPath, mime_content_type($targetPath), $originalName);
$postFields = ['uploaded_file' => $cfile];

curl_setopt_array($curl, [
    CURLOPT_URL => $backendUrl,
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POSTFIELDS => $postFields
]);

$response = curl_exec($curl);
$httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
$error = curl_error($curl);
curl_close($curl);

if ($error) {
    echo json_encode(["error" => "Failed to forward to backend: $error"]);
    exit;
}

if ($httpCode !== 200) {
    echo json_encode(["error" => "Backend returned HTTP $httpCode", "raw_response" => $response]);
    exit;
}

$json = json_decode($response, true);
if ($json === null) {
    echo json_encode(["error" => "Backend returned invalid JSON", "raw_response" => $response]);
    exit;
}

echo json_encode($json);
exit;



?>
