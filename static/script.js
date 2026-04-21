// --- CITY WEATHER LOGIC ---
function toggleCityInput() {
    const container = document.getElementById('cityInputContainer');
    if (container) {
        container.style.display = container.style.display === 'none' ? 'block' : 'none';
    }
}

function updateCity() {
    const newCity = document.getElementById('newCityInput').value.trim();
    if (newCity) {
        window.location.href = `/?city=${encodeURIComponent(newCity)}`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // Update hidden city input if it exists
    const params = new URLSearchParams(window.location.search);
    const cityParam = params.get('city');
    if (cityParam) {
        const hiddenInput = document.getElementById('hiddenCityInput');
        if (hiddenInput) hiddenInput.value = cityParam;
    }

    
    // File Upload Logic
    const fileDropArea = document.querySelector(".file-drop-area");
    const fileInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("imagePreviewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const form = document.getElementById("uploadForm");
    const analyzeBtn = document.getElementById("analyzeBtn");

    if (fileDropArea && fileInput) {
        
        fileDropArea.addEventListener("dragover", (e) => {
            e.preventDefault();
            fileDropArea.classList.add("dragover");
        });

        fileDropArea.addEventListener("dragleave", () => {
            fileDropArea.classList.remove("dragover");
        });

        fileDropArea.addEventListener("drop", (e) => {
            e.preventDefault();
            fileDropArea.classList.remove("dragover");
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                showPreview(fileInput.files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                showPreview(fileInput.files[0]);
            }
        });

        function showPreview(file) {
            if (file && file.type.startsWith("image/")) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    previewContainer.style.display = "block";
                    fileDropArea.querySelector(".file-message").innerHTML = `<b>${file.name}</b> selected!`;
                };
                reader.readAsDataURL(file);
            }
        }

        form.addEventListener("submit", (e) => {
            if(fileInput.files.length === 0) {
                e.preventDefault();
                alert("Please select an image first!");
                return;
            }
            // Add loading spinner animation
            analyzeBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...';
            analyzeBtn.disabled = true;
            analyzeBtn.style.opacity = "0.8";
            analyzeBtn.style.animation = "none"; // Stop pulse while loading
        });
    }

    // Confidence Bar Animation
    const progressBarFill = document.querySelector(".progress-bar-fill");
    if (progressBarFill) {
        // Get the target width from a custom data attribute if available, or inline style
        const targetWidth = progressBarFill.getAttribute("data-width") || progressBarFill.style.width;
        // Start at 0
        progressBarFill.style.width = "0%";
        // Animate to target width after a slight delay
        setTimeout(() => {
            progressBarFill.style.width = targetWidth;
        }, 300);
    }

    // Staggered Entrance Animations using Intersection Observer
    const animatedElements = document.querySelectorAll('.animate-fade-up');
    
    if (animatedElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Start the animation
                    entry.target.style.animationPlayState = 'running';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        animatedElements.forEach(el => {
            // Pause animation initially
            el.style.animationPlayState = 'paused';
            observer.observe(el);
        });
    }

    // Chatbot Logic
    const chatbotWidget = document.getElementById("chatbot-widget");
    const chatbotHeader = document.getElementById("chatbot-header");
    const chatbotToggleIcon = document.getElementById("chatbot-toggle-icon");
    const chatInput = document.getElementById("chat-input");
    const chatSendBtn = document.getElementById("chat-send-btn");
    const chatMessages = document.getElementById("chat-messages");

    if (chatbotWidget) {
        chatbotHeader.addEventListener("click", () => {
            chatbotWidget.classList.toggle("collapsed");
            if (chatbotWidget.classList.contains("collapsed")) {
                chatbotToggleIcon.classList.remove("fa-chevron-down");
                chatbotToggleIcon.classList.add("fa-chevron-up");
            } else {
                chatbotToggleIcon.classList.remove("fa-chevron-up");
                chatbotToggleIcon.classList.add("fa-chevron-down");
                chatInput.focus();
            }
        });

        const sendMessage = async () => {
            const message = chatInput.value.trim();
            if (!message) return;

            // Append user message
            const userMsgDiv = document.createElement("div");
            userMsgDiv.className = "chat-message user";
            userMsgDiv.innerText = message;
            chatMessages.appendChild(userMsgDiv);
            chatInput.value = "";
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Append "Typing..."
            const typingDiv = document.createElement("div");
            typingDiv.className = "chat-message bot";
            typingDiv.innerHTML = '<i class="fa-solid fa-ellipsis fa-fade"></i>';
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                
                chatMessages.removeChild(typingDiv);
                
                const botMsgDiv = document.createElement("div");
                botMsgDiv.className = "chat-message bot";
                // format simple markdown asterisks to bold if present
                let formattedText = (data.response || data.error).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
                botMsgDiv.innerHTML = formattedText;
                chatMessages.appendChild(botMsgDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } catch (error) {
                chatMessages.removeChild(typingDiv);
                const errorDiv = document.createElement("div");
                errorDiv.className = "chat-message bot";
                errorDiv.innerText = "Error connecting to the AI Assistant.";
                chatMessages.appendChild(errorDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        };

        chatSendBtn.addEventListener("click", sendMessage);
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                sendMessage();
            }
        });
    }

    // Live Camera Logic
    const openCameraBtn = document.getElementById("openCameraBtn");
    const closeCameraBtn = document.getElementById("closeCameraBtn");
    const captureBtn = document.getElementById("captureBtn");
    const cameraContainer = document.getElementById("camera-container");
    const cameraVideo = document.getElementById("camera-video");
    const cameraCanvas = document.getElementById("camera-canvas");
    const fileDropAreaEl = document.getElementById("file-drop-area");
    const cameraActionsContainer = document.getElementById("cameraActionsContainer");

    let stream = null;

    if (openCameraBtn) {
        openCameraBtn.addEventListener("click", async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
                cameraVideo.srcObject = stream;
                cameraContainer.style.display = "flex";
                fileDropAreaEl.style.display = "none";
                cameraActionsContainer.style.display = "none";
                previewContainer.style.display = "none";
            } catch (err) {
                alert("Could not access camera. Please check permissions.");
                console.error("Camera error:", err);
            }
        });

        const stopCamera = () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            cameraContainer.style.display = "none";
            fileDropAreaEl.style.display = "block";
            cameraActionsContainer.style.display = "block";
        };

        closeCameraBtn.addEventListener("click", stopCamera);

        captureBtn.addEventListener("click", () => {
            cameraCanvas.width = cameraVideo.videoWidth;
            cameraCanvas.height = cameraVideo.videoHeight;
            const ctx = cameraCanvas.getContext("2d");
            ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
            
            // Convert to Blob and then to File object
            cameraCanvas.toBlob((blob) => {
                const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                
                // Assign File to the hidden file input using DataTransfer
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                
                // Show preview
                showPreview(file);
                stopCamera();
            }, "image/jpeg", 0.9);
        });
    }

});
