<?php
session_start();

if (!isset($_SESSION['session_id'])) {
    http_response_code(400);
    echo json_encode(["error" => "Session not initialized"]);
    exit();
}

$session_id = $_SESSION['session_id'];
$body = file_get_contents("php://input");
$data = json_decode($body, true);

if (!isset($data['history_html'])) {
    http_response_code(400);
    echo json_encode(["error" => "Missing history"]);
    exit();
}

$filename = "session_history/history_{$session_id}.html";
file_put_contents($filename, $data['history_html']);

echo json_encode(["status" => "saved"]);
?>
