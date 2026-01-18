
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("renewSession").addEventListener("click", async function() {
        try {
            let response = await fetch("index.php?renew=1");
            let data = await response.json();

            if (data.status === "success") {
                document.getElementById("session-id-text").innerText = data.session_id;
                document.getElementById("session_id").value = data.session_id;
            }
        } catch (error) {
            console.error("Error renewing session:", error);
        }
    });
});

function addCopyButtons() {
    console.log("Checking for <pre> elements to add copy buttons...");

    document.querySelectorAll("pre").forEach(pre => {
        if (pre.querySelector(".copy-button")) return; // Avoid duplicate buttons

        console.log("Adding copy button to:", pre);

        let code = pre.querySelector("code");
        if (!code) {
            console.warn("No <code> block inside <pre>, skipping...");
            return;
        }

        let button = document.createElement("button");
        button.innerText = "📋 Copy";
        button.classList.add("copy-button");

        pre.style.position = "relative";
        pre.appendChild(button);

        button.addEventListener("click", async function () {
            let textToCopy = code.innerText.trim(); // Ensure no extra spaces

            console.log("Copying text:", textToCopy); // Debugging output

            try {
                await navigator.clipboard.writeText(textToCopy);
                console.log("✅ Successfully copied:", textToCopy);
                button.innerText = "✅ Copied!";
                setTimeout(() => (button.innerText = "📋 Copy"), 1500);
            } catch (err) {
                console.error("❌ Failed to copy:", err);
                button.innerText = "❌ Copy Failed";
                setTimeout(() => (button.innerText = "📋 Copy"), 1500);
            }
        });
    });
}

document.addEventListener("DOMContentLoaded", addCopyButtons);

document.getElementById("queryForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let queryInput = document.getElementById("queryInput");
    let query = queryInput.value.trim();
    let sessionId = document.getElementById("session_id").value;
    let model = document.getElementById("modelSelect").value;
    let responseElement = document.getElementById("response");

    if (!query) return;

    // ✅ Save current request & response into history before submitting a new one
    let previousQuery = document.getElementById("request").innerHTML;
    let previousResponse = document.getElementById("response").innerHTML;

    if (previousQuery.trim() && previousResponse.trim()) {
        let historyDiv = document.getElementById("history");
        let newEntry = document.createElement("div");
        newEntry.style.marginBottom = "20px";
        newEntry.innerHTML = `
            <div>${previousQuery}</div>
            <div>${previousResponse}</div>
            <hr>
        `;
        historyDiv.appendChild(newEntry);
		historyDiv.scrollTop = historyDiv.scrollHeight;	
    }


    document.getElementById("response-container").style.display = "block";
    document.getElementById("request").innerHTML = "<strong>You asked:</strong> " + query;
    responseElement.innerHTML = "<em>Waiting for response...</em>";

    marked.setOptions({
        langPrefix: 'language-', // highlight.js expects this class prefix
        highlight: function(code, lang) {
            return hljs.highlightAuto(code).value;
        }
    });

    function renderMarkdown() {
        const html = marked.parse(mybufr2);
        responseElement.innerHTML = html;

        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
        addCopyButtons();
    }

    try {
		queryInput.value = "Waiting for DAWID to respond ..."
		queryInput.style.backgroundColor = "#e0e0e0";
		queryInput.disabled = true;

        let response = await fetch("process_query.php", {
            method: "POST",
            body: new URLSearchParams({
                q: query,
                session_id: sessionId,
                model: model
            })
        });

        if (!response.ok) {
            throw new Error("Network response was not ok.");
        }

        let reader = response.body.getReader();
        let decoder = new TextDecoder("utf-8");

        let receivedChunks = new Set();  // ✅ Prevent duplicate responses
        responseElement.innerHTML = "";  // Clear old response
        mybufr = "";
        counter = 0;
		const init_response = "One moment ...";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            let chunk = decoder.decode(value, { stream: true });

            console.log("chunk:", chunk);
            try {
                let lines = chunk.trim().split("\n");
                lines.forEach(line => {
                    let data = JSON.parse(line);
                    mybufr += data.response;

                    // (Your original conditional code block is kept commented)
                    /*
                    if (data.response && !receivedChunks.has(data.response)) {
                        ...
                    }
                    */
                });

                console.log("mybufr:", mybufr);

				if (mybufr.length > init_response.length && mybufr.startsWith(init_response)) {
					mybufr = mybufr.substring(init_response.length);
				}

                let mybufr2 = marked.parse(mybufr);
                responseElement.innerHTML = mybufr2;

                document.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });

                addCopyButtons();

                // Optional renderMarkdown call:
                // renderMarkdown();

                console.log("innerHTML:", responseElement.innerHTML);

            } catch (e) {
                console.error("Invalid JSON chunk received:", chunk);
            }
        }

		queryInput.style.backgroundColor = "white";
        queryInput.value = ""; // Clear input field
		queryInput.disabled = false;


        fetch("save_response.php", {
            method: "POST",
            body: new URLSearchParams({ text: responseElement.innerHTML })
        });

    } catch (error) {
        responseElement.innerHTML = "Error: " + error.message;
    }
});
