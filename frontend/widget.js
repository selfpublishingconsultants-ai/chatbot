(function() {
    // 1. Detect Host URL from script src, default to local origin
    const scriptElement = document.currentScript;
    let hostUrl = window.location.origin;
    if (scriptElement && scriptElement.src) {
        try {
            const url = new URL(scriptElement.src);
            hostUrl = url.origin;
        } catch (e) {
            console.warn("[SPC Widget] Failed to parse script origin, defaulting to local origin.");
        }
    }

    // Prevent duplicate injection
    if (window.__spc_widget_injected) return;
    window.__spc_widget_injected = true;

    // 2. Inject CSS Styles for Bubble and Iframe
    const style = document.createElement('style');
    style.innerHTML = `
        /* Floating Chat Bubble Button */
        #spc-chat-bubble {
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
            box-shadow: 0 4px 16px rgba(217, 119, 6, 0.4);
            cursor: pointer;
            z-index: 999999;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        #spc-chat-bubble:hover {
            transform: scale(1.08) translateY(-2px);
            box-shadow: 0 6px 20px rgba(217, 119, 6, 0.6);
        }
        #spc-chat-bubble:active {
            transform: scale(0.95);
        }
        #spc-chat-bubble svg {
            width: 26px;
            height: 26px;
            fill: white;
            transition: transform 0.3s ease;
        }
        #spc-chat-bubble.active svg {
            transform: rotate(90deg);
        }

        /* Iframe Container */
        #spc-chat-container {
            position: fixed;
            bottom: 96px;
            right: 24px;
            width: 380px;
            height: 580px;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            z-index: 999998;
            overflow: hidden;
            display: none;
            opacity: 0;
            transform: translateY(20px) scale(0.95);
            transition: opacity 0.3s ease, transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            background: #0b0f19;
        }
        #spc-chat-container.active {
            display: block;
            opacity: 1;
            transform: translateY(0) scale(1);
        }
        #spc-chat-iframe {
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }

        /* Mobile Responsive adjustments */
        @media (max-width: 480px) {
            #spc-chat-container {
                width: 100% !important;
                height: 100% !important;
                bottom: 0 !important;
                right: 0 !important;
                border-radius: 0 !important;
                border: none !important;
            }
            #spc-chat-bubble.active {
                bottom: 12px !important;
                right: 12px !important;
                width: 50px !important;
                height: 50px !important;
                box-shadow: none !important;
            }
        }
    `;
    document.head.appendChild(style);

    // 3. Create Iframe Container
    const container = document.createElement('div');
    container.id = 'spc-chat-container';
    
    const iframe = document.createElement('iframe');
    iframe.id = 'spc-chat-iframe';
    // Do not set source immediately for lazy loading optimization
    iframe.dataset.src = `${hostUrl}/widget`;
    container.appendChild(iframe);
    document.body.appendChild(container);

    // 4. Create Chat Bubble Button
    const bubble = document.createElement('div');
    bubble.id = 'spc-chat-bubble';
    // Chat Icon SVG
    bubble.innerHTML = `
        <svg id="spc-bubble-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <!-- Normal Chat Icon -->
            <path class="normal-path" d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
        </svg>
    `;
    document.body.appendChild(bubble);

    // Close Icon SVG to swap when active
    const closeIconSVG = `
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
    `;
    const openIconSVG = `
        <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
    `;

    // 5. Toggle Toggle Logic
    let isIframeLoaded = false;
    bubble.addEventListener('click', function() {
        const isOpened = container.classList.contains('active');
        const iconSvg = document.getElementById('spc-bubble-icon');

        if (!isOpened) {
            // Lazy load the iframe source
            if (!isIframeLoaded) {
                iframe.src = iframe.dataset.src;
                isIframeLoaded = true;
            }
            container.style.display = 'block';
            // Trigger animation in next frame
            setTimeout(() => {
                container.classList.add('active');
                bubble.classList.add('active');
                iconSvg.innerHTML = closeIconSVG;
            }, 10);
        } else {
            container.classList.remove('active');
            bubble.classList.remove('active');
            iconSvg.innerHTML = openIconSVG;
            // Hide container after animation finishes
            setTimeout(() => {
                if (!container.classList.contains('active')) {
                    container.style.display = 'none';
                }
            }, 300);
        }
    });
})();
