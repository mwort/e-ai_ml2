document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("uploadFile");
    const uploadResult = document.getElementById("upload-result");
    const requestDiv = document.getElementById("request");
    const responseDiv = document.getElementById("response");

    // 🔽 Submit form automatically when a file is picked
    fileInput.addEventListener("change", function() {
        if (fileInput.files.length) {
            uploadForm.requestSubmit();  // triggers the submit event
        }
    });

    uploadForm.addEventListener("submit", function (e) {
        e.preventDefault();
        if (!fileInput.files.length) return;

        const filename = fileInput.files[0].name;
        const formData = new FormData();
        formData.append("uploaded_file", fileInput.files[0]);
        formData.append("request", "Upload file " + filename);

        uploadResult.innerText = `⏳ Uploading ${filename}...`;
        console.log("📁 Starting upload for:", filename);

        fetch("upload_handler.php", {
            method: "POST",
            body: formData,
        })
        .then(response => {
            console.log("📥 Got response from upload_handler.php");
            return response.json();
        })
        .then(data => {
            const output = data.message || data.error || JSON.stringify(data);
            uploadResult.innerText = `✅ Upload result: ${output}`;
            requestDiv.innerText = "Upload file " + filename;
            responseDiv.innerText = output;
        })
        .catch(err => {
            console.error("❌ Upload failed:", err);
            uploadResult.innerText = "❌ Upload error: " + err;
        });
    });
});
