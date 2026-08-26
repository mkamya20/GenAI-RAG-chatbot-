// Relative URL — HTML is served by FastAPI at the same origin
const API_BASE_URL = '';

// DOM elements
const chatInput = document.getElementById('chatInput');
const chatButton = document.getElementById('chatButton');
const chatMessages = document.getElementById('chatMessages');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

// Event listeners
chatButton.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});

// Check API status on load
checkStatus();

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (!response.ok) throw new Error('API error');
        
        const data = await response.json();
        
        if (data.status === 'ok') {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'Connected';
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Disconnected';
        }
    } catch (error) {
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'Cannot connect to API';
    }
}

function createMessageElement(role, text, renderMarkdown = false) {
    const message = document.createElement('div');
    message.className = `message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (renderMarkdown && typeof marked !== 'undefined') {
        const html = marked.parse(text);
        bubble.innerHTML = typeof DOMPurify !== 'undefined' 
            ? DOMPurify.sanitize(html) 
            : html;
    } else {
        bubble.textContent = text;
    }
    
    message.appendChild(bubble);
    return message;
}

function appendSourcesElement(messageEl, sources) {
    if (!sources || sources.length === 0) return;
    
    // Deduplicate and group pages by filename, preserve URL
    const grouped = {};
    sources.forEach(s => {
        const filename = s.filename;
        if (!grouped[filename]) {
            grouped[filename] = {
                pages: new Set(),
                url: s.url || null
            };
        }
        if (s.page_numbers && s.page_numbers.length > 0) {
            s.page_numbers.forEach(p => grouped[filename].pages.add(p));
        }
    });
    
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';
    
    const label = document.createElement('strong');
    label.textContent = 'Sources:';
    sourcesDiv.appendChild(label);
    
    const list = document.createElement('ul');
    list.className = 'sources-list';
    
    Object.entries(grouped).forEach(([filename, info]) => {
        const item = document.createElement('li');
        const pagesArray = Array.from(info.pages).sort((a, b) => a - b);
        const isPdf = filename.toLowerCase().endsWith('.pdf');
        const hasUrl = info.url && info.url.startsWith('http');
        
        if (isPdf) {
            const link = document.createElement('a');
            link.href = `${API_BASE_URL}/api/pdfs/${encodeURIComponent(filename)}/download`;
            link.target = '_blank';
            link.textContent = filename;
            item.appendChild(link);
        } else if (hasUrl) {
            const link = document.createElement('a');
            link.href = info.url;
            link.target = '_blank';
            link.textContent = filename;
            item.appendChild(link);
        } else {
            item.appendChild(document.createTextNode(filename));
        }
        
        if (pagesArray.length > 0) {
            item.appendChild(document.createTextNode(` (Page ${pagesArray.join(', ')})`));
        }
        
        list.appendChild(item);
    });
    
    sourcesDiv.appendChild(list);
    messageEl.appendChild(sourcesDiv);
}

async function sendChatMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.disabled = true;
    chatButton.disabled = true;
    chatInput.value = '';

    // Add user message
    const userMessage = createMessageElement('user', query);
    chatMessages.appendChild(userMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Add loading message
    const loadingMessage = createMessageElement('assistant', 'Thinking...');
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 5, use_rag: true })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        // Replace loading message with response
        loadingMessage.remove();
        
        const assistantMessage = createMessageElement('assistant', data.answer, true);
        appendSourcesElement(assistantMessage, data.sources);
        chatMessages.appendChild(assistantMessage);

    } catch (error) {
        loadingMessage.remove();
        
        const errorMessage = document.createElement('div');
        errorMessage.className = 'message assistant';
        
        const errorBubble = document.createElement('div');
        errorBubble.className = 'message-bubble error';
        errorBubble.textContent = `Error: ${error.message}`;
        
        errorMessage.appendChild(errorBubble);
        chatMessages.appendChild(errorMessage);
    } finally {
        chatMessages.scrollTop = chatMessages.scrollHeight;
        chatInput.disabled = false;
        chatButton.disabled = false;
        chatInput.focus();
    }
}