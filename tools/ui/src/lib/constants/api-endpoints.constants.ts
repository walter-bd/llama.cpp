export const API_MODELS = {
	LIST: '/v1/models',
	LOAD: '/models/load',
	SSE: '/models/sse',
	UNLOAD: '/models/unload'
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const API_CHAT = {
	COMPLETIONS: './v1/chat/completions',
	CONTROL: './v1/chat/completions/control'
};

// Plastic sidecar lifecycle and observation routes. The sidecar remains an
// explicit opt-in extension to normal chat requests.
export const API_SIDECAR = {
	BASE: './v1/sidecar',
	CREATE: './v1/sidecar/create',
	LEARNING: './v1/sidecar/learning',
	LOAD: './v1/sidecar/load',
	OBSERVE: './v1/sidecar/observe',
	OBSERVE_SEQUENCE: './v1/sidecar/observe_sequence',
	RESET: './v1/sidecar/reset',
	SAVE: './v1/sidecar/save',
	STATUS: './v1/sidecar/status',
	UNLOAD: './v1/sidecar/unload'
};

export const API_NATIVE_COMPLETION = './completion';
export const API_TOKENIZE = './tokenize';

// slot introspection, requires the --slots flag on the server
export const API_SLOTS = {
	LIST: './slots'
};

export const API_TOOLS = {
	EXECUTE: '/tools',
	LIST: '/tools'
};

// resumable stream routes, the conv::model identity travels as the conv_id query param
// because model names can contain slashes that a path segment cannot carry
// resume retry cadence while the owning model is still loading (server answers 503)
export const STREAM_RESUME_RETRY_MS = 2000;

export const API_STREAM = {
	BASE: './v1/stream',
	LOOKUP: './v1/streams/lookup'
};

/** CORS proxy endpoint path */
export const CORS_PROXY_ENDPOINT = '/cors-proxy';
