<?php
header("Content-Type: application/json");
header("Cache-Control: no-cache");
header("Connection: keep-alive");
ob_implicit_flush(true);

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['q'], $_POST['session_id'], $_POST['model'])) {
    $q = escapeshellarg($_POST["q"]);
    $session_id = escapeshellarg($_POST["session_id"]);
    $model = escapeshellarg($_POST["model"]);

    // Ensure session responses are cleared before streaming new output
    session_start();
    $_SESSION["last_response"] = ""; // Reset last response

    $command = "python3 ollama_fetch.py $q $session_id $model 2>&1";
    $handle = popen($command, "r");

    // Open file to store response
    $file = fopen("response_from_query.txt", "a"); // Append mode

    if ($handle && $file) {
        while (!feof($handle)) {
            $chunk = fgets($handle, 1024);
            if ($chunk) {
                $_SESSION["last_response"] .= $chunk; // Store new response
                echo $chunk;
                ob_flush();
                flush();

                // Write chunk to file
                fwrite($file, $chunk);
            }
        }
        pclose($handle);
        fclose($file); // Close file after writing
    }
} else {
    echo json_encode(["error" => "Invalid request"]);
    flush();
}
?>
