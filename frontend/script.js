const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const exampleBtns = document.querySelectorAll('.example-btn');

const API_URL = 'http://localhost:8000/chat';

const chatContainer = document.querySelector('.chat-container');

function appendMessage(text, role, officialLinks = []) {
    // Hide welcome screen and adjust layout on first message
    if (!chatContainer.classList.contains('active')) {
        chatContainer.classList.add('active');
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    // Replace newlines with <br>
    let formattedText = text.replace(/\n/g, '<br>');

    // Add multiple official links as buttons
    if (officialLinks && officialLinks.length > 0 && role === 'assistant') {
        formattedText += `<div class="link-container">`;
        officialLinks.forEach(link => {
            formattedText += `<a href="${link.url}" target="_blank" class="source-link">${link.label}</a>`;
        });
        formattedText += `</div>`;
    }

    messageDiv.innerHTML = formattedText;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function handleSendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    // Clear input and add user message
    userInput.value = '';
    appendMessage(query, 'user');

    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant-message loading';
    loadingDiv.innerHTML = 'Searching official HDFC files...';
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: query })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server Error (Status ${response.status})`);
        }

        const data = await response.json();

        // Remove loading and add assistant message
        chatBox.removeChild(loadingDiv);
        appendMessage(data.answer, 'assistant', data.official_links);

    } catch (error) {
        if (loadingDiv.parentNode) chatBox.removeChild(loadingDiv);
        appendMessage(`Error: ${error.message}`, 'assistant');
        console.error('Error:', error);
    }
}

sendBtn.addEventListener('click', handleSendMessage);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSendMessage();
});

exampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        userInput.value = btn.getAttribute('data-query');
        handleSendMessage();
    });
});
