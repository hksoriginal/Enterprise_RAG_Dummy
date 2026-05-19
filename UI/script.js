document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    
    // API URL to backend
    const API_URL = 'http://localhost:8000/rag/get-answer';

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = userInput.value.trim();
        if (!query) return;

        // Add user message to UI
        addMessage(query, 'user');
        
        // Clear input field
        userInput.value = '';
        
        // Add loading indicator
        const loadingId = addLoadingIndicator();
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();
            
            // Remove loading indicator
            removeElement(loadingId);
            
            if (response.ok) {
                // The API response might have different structures
                // We'll try to extract the answer from common RAG response structures
                const answer = data.answer || data.response || data.result || data.text || JSON.stringify(data);
                addMessage(answer, 'system');
                console.log("Full response:", data);
            } else {
                // Try to extract the error message more robustly
                let errorDetail = data.detail;
                if (typeof errorDetail === 'object' && errorDetail !== null) {
                    // If detail is an object, try to find the message
                    if (errorDetail.message) {
                        errorDetail = errorDetail.message;
                    } else if (errorDetail.detail) {
                        errorDetail = errorDetail.detail;
                    } else {
                        // Fallback to stringifying the object
                        errorDetail = JSON.stringify(errorDetail);
                    }
                }
                
                addMessage(`Error: ${errorDetail || 'An error occurred while processing the request'}`, 'system');
            }
        } catch (error) {
            removeElement(loadingId);
            addMessage('Error: Failed to connect to the backend server. Make sure the API is running on port 8000.', 'system');
            console.error('API Error:', error);
        }
    });

    function addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const iconClass = sender === 'user' ? 'fa-user' : 'fa-robot';
        
        // Parse basic markdown if present (bold, newlines)
        let formattedContent;
        if (typeof content === 'string') {
            formattedContent = content
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
        } else {
            // Fallback for objects
            formattedContent = `<pre style="font-family: monospace; white-space: pre-wrap; font-size: 0.85rem; overflow-x: auto;">${JSON.stringify(content, null, 2)}</pre>`;
        }
            
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fa-solid ${iconClass}"></i>
            </div>
            <div class="message-content">
                <p>${formattedContent}</p>
            </div>
        `;
        
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.id = id;
        messageDiv.className = `message system-message`;
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        chatHistory.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
