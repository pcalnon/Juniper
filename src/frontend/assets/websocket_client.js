/**
 * Juniper Canopy - WebSocket Client
 *
 * Provides real-time push updates from the backend via WebSocket,
 * replacing HTTP polling for efficient real-time monitoring.
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Type-based message dispatching
 * - Connection status tracking
 * - In-memory message buffering for Dash integration
 */

class CascorWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.handlers = {};
        this.statusHandlers = [];
        this.messageBuffer = [];
        this.status = 'disconnected';
        this.reconnectAttempts = 0;
        this.maxReconnectDelay = 5000; // 5 seconds max backoff
        this.baseReconnectDelay = 500; // 500ms initial delay

        // Auto-connect on construction
        this.connect();
    }

    /**
     * Establish WebSocket connection
     */
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            console.log('[CascorWS] Already connected or connecting');
            return;
        }

        this._setStatus('connecting');
        console.log(`[CascorWS] Connecting to ${this.url}`);

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('[CascorWS] Connected');
                this._setStatus('open');
                this.reconnectAttempts = 0; // Reset on successful connection
            };

            this.ws.onclose = (event) => {
                console.log(`[CascorWS] Disconnected: ${event.code} ${event.reason}`);
                this._setStatus('closed');
                this._scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('[CascorWS] Error:', error);
                this._setStatus('error');
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this._handleMessage(message);
                } catch (err) {
                    console.error('[CascorWS] Failed to parse message:', err);
                }
            };
        } catch (err) {
            console.error('[CascorWS] Connection failed:', err);
            this._setStatus('error');
            this._scheduleReconnect();
        }
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= 10) {
            console.warn('[CascorWS] Max reconnection attempts reached. Stopping.');
            return;
        }

        const delay = Math.min(
            this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );

        this.reconnectAttempts++;
        console.log(`[CascorWS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => this.connect(), delay);
    }

    /**
     * Register handler for specific message type
     * @param {string} type - Message type to handle
     * @param {function} handler - Handler function(data)
     */
    on(type, handler) {
        this.handlers[type] = handler;
    }

    /**
     * Register handler for connection status changes
     * @param {function} handler - Handler function(status)
     */
    onStatus(handler) {
        this.statusHandlers.push(handler);
    }

    /**
     * Handle incoming message
     * @private
     */
    _handleMessage(message) {
        const { type, data } = message;

        // Add to buffer for Dash clientside callbacks
        this.messageBuffer.push({ type, data, timestamp: Date.now() });

        // Limit buffer size (keep last 100 messages)
        if (this.messageBuffer.length > 100) {
            this.messageBuffer.shift();
        }

        // Dispatch to registered handler
        if (this.handlers[type]) {
            try {
                this.handlers[type](data);
            } catch (err) {
                console.error(`[CascorWS] Handler error for type '${type}':`, err);
            }
        }
    }

    /**
     * Set connection status and notify handlers
     * @private
     */
    _setStatus(status) {
        this.status = status;
        this.statusHandlers.forEach(handler => {
            try {
                handler(status);
            } catch (err) {
                console.error('[CascorWS] Status handler error:', err);
            }
        });
    }

    /**
     * Get and clear buffered messages (for Dash integration)
     * @returns {Array} Buffered messages
     */
    getBufferedMessages() {
        const messages = [...this.messageBuffer];
        this.messageBuffer = [];
        return messages;
    }

    /**
     * Send message to server (for control commands)
     * @param {object} message - Message to send
     * @returns {Promise} Promise that resolves when ack is received or rejects on timeout
     */
    send(message) {
        return new Promise((resolve, reject) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(message));

                // Set up ack handler with timeout
                const timeout = setTimeout(() => {
                    reject(new Error('Command timeout (no ack received)'));
                }, 5000);

                // Listen for ack
                const ackHandler = (data) => {
                    if (data.command === message.command) {
                        clearTimeout(timeout);
                        this.off('control_ack', ackHandler);
                        resolve(data);
                    }
                };

                this.on('control_ack', ackHandler);
            } else {
                reject(new Error('WebSocket not connected'));
                console.warn('[CascorWS] Cannot send message: not connected');
            }
        });
    }

    /**
     * Remove handler for specific message type
     * @param {string} type - Message type
     * @param {function} handler - Handler function to remove
     */
    off(type, handler) {
        if (this.handlers[type] === handler) {
            delete this.handlers[type];
        }
    }

    /**
     * Close connection
     */
    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Create global singleton WebSocket for training updates
const trainingWSUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/training`;
window.cascorWS = new CascorWebSocket(trainingWSUrl);

// Create global singleton WebSocket for control commands
const controlWSUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/control`;
window.cascorControlWS = new CascorWebSocket(controlWSUrl);

// Log status changes
window.cascorWS.onStatus(status => console.log(`[Training WS] Status: ${status}`));
window.cascorControlWS.onStatus(status => console.log(`[Control WS] Status: ${status}`));

console.log('[CascorWS] WebSocket clients initialized');
