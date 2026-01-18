<?php
session_start();

// Redirect if not logged in
if (!isset($_SESSION['logged_in']) || $_SESSION['logged_in'] !== true) {
    header("Location: login.php");
    exit();
}

// ✅ Handle session renewal request
if (isset($_GET['renew'])) {
    session_regenerate_id(true); // ✅ Regenerate session ID securely
    $_SESSION['session_id'] = session_id();
    echo json_encode(["status" => "success", "session_id" => $_SESSION['session_id']]);
    exit();
}

// Ensure session ID is set
if (!isset($_SESSION['session_id']) || empty($_SESSION['session_id'])) {
    $_SESSION['session_id'] = session_id() ?: uniqid();
}
$session_id = $_SESSION['session_id'];

// Define available models
$models = [
    "llama3:8b" => "llama3:8b",
    "gpt-4o-mini" => "gpt-4o-mini"
];
$default_model = "gpt-4o-mini"; // Default model selection
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DWD AI Interface</title>
	<link rel="stylesheet" href="dawid_style.css">

    <!-- ✅ marked.js for Markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <!-- ✅ highlight.js full bundle with all languages -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
	<script src="dawid_upload.js"></script>
	</head>
<body>
    <div class="container">
        <input type="hidden" id="session_id" name="session_id" value="<?php echo htmlspecialchars($session_id); ?>">
        <div class="session-id-container">
            <span>Session ID: <span id="session-id-text"><?php echo htmlspecialchars($session_id); ?></span></span>
            <button id="renewSession" class="renew-session">↻</button>
        </div>
<!-- ============================================================================================== -->
<!-- Website Contents                                                                               -->
<!-- ============================================================================================== -->
        <h1 class="main-heading">D<sup>A</sup>W<sup>I</sup>D</h1>

        <div id="response-container">
            <div id="history"></div>
            <div id="request"></div>
            <div id="response"></div>
        </div>

		<form id="uploadForm" enctype="multipart/form-data" style="display: none;">
			<input type="file" id="uploadFile" name="uploaded_file" />
		</form>

<!-- Visible label somewhere else on the page -->
		<form id="queryForm">
			<div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
				<!-- 📎 Upload icon triggers hidden file input -->

				<!-- Model selection -->
				<label for="uploadFile" title="Upload file" style="cursor: pointer; font-size: 20px;">📁</label>
				<label for="modelSelect">Select Model:</label>
				<select id="modelSelect" name="model">
					<?php foreach ($models as $key => $name): ?>
						<option value="<?php echo htmlspecialchars($key); ?>" <?php echo ($key == $default_model) ? "selected" : ""; ?>>
							<?php echo htmlspecialchars($name); ?>
						</option>
					<?php endforeach; ?>
				</select>
			</div>
			<br>
			<div id="upload-result" style="display: none;"></div>
			<textarea id="queryInput" name="q" placeholder="Enter your question..." required></textarea><br>
			<button type="submit">Ask</button>
		</form>

    </div>
    <script src="display.js"></script> <!-- External JavaScript file -->
</body>
</html>
