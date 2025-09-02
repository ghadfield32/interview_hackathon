// web/src/services/api.js
import toast from 'react-hot-toast';     // add toast here for 401 handler

/**
 * API base resolution:
 *  - Single authoritative VITE_API_URL comes from generated web/.env
 *    (derived from LOCAL/STAGING/RAILWAY base keys in root config.yaml).
 *  - We still keep a build-time constant (__BUILD_API_URL__) as a fallback.
 *  - Only VITE_* keys are exposed to client code (per Vite rules).
 */
const API_BASE_URL = (() => {
  const envURL   = import.meta.env?.VITE_API_URL || ''
  const buildURL = (typeof __BUILD_API_URL__ !== 'undefined' && __BUILD_API_URL__) || ''
  const chosenRaw = envURL || buildURL

  let base = chosenRaw.replace(/\/+$/, '')
  if (base && !/\/api\/v1$/.test(base)) {
    base = base + '/api/v1'
  }

  if (!base) {
    console.warn('[ApiService] No VITE_API_URL provided – falling back to /api/v1 (DEV ONLY)')
    return '/api/v1'
  }

  console.log('[ApiService] Resolved API base:', {
    envURL, buildURL, final: base
  })

  return base
})();

// ─────────── Helper to join paths safely ─────────────────────────
const join = (base, path) => {
  const normalBase = base.replace(/\/+$/, '');     // trim trailing /
  const normalPath = path.replace(/^\/+/, '');     // trim leading /
  return `${normalBase}/${normalPath}`;            // single slash in-between
};

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;            // now correct
    this.defaultHeaders = {};  // Remove default Content-Type to avoid CORS preflight for GETs
  }

  async request(endpoint, options = {}) {
    // Avoid repeating /api/v1 if caller passes it
    const cleanEndpoint = endpoint.replace(/^\/?api\/v1\//, '')
    const url = join(this.baseURL, cleanEndpoint)
    const token = localStorage.getItem('jwt')

    console.debug('[API Request]', {
      method: options.method || 'GET',
      endpoint,
      cleanEndpoint,
      url
    })

    const cfg = {
      method: options.method || 'GET',
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers
      }
    };

    try {
      const res = await fetch(url, cfg)

      const rateLimitRemaining = res.headers.get('X-RateLimit-Remaining')
      const rateLimitLimit = res.headers.get('X-RateLimit-Limit')
      const retryAfter = res.headers.get('Retry-After')

      if (rateLimitRemaining !== null) {
        const remaining = parseInt(rateLimitRemaining, 10)
        const limit = parseInt(rateLimitLimit, 10)
        if (remaining <= 3 && remaining > 0) {
          toast.warning(`Rate limit warning: ${remaining}/${limit} remaining`)
        }
        console.debug(`[API] Rate limit: ${remaining}/${limit}`)
      }

      if (res.status === 401) {
        localStorage.removeItem('jwt')
        toast.error('Session expired – please log in again.')
        window.location.replace('/login')
        return
      }

      if (res.status === 429) {
        const retrySeconds = retryAfter ? parseInt(retryAfter, 10) : 60
        toast.error(`Rate limit exceeded. Wait ${retrySeconds}s.`)
        throw new Error(`429 retry after ${retrySeconds}s`)
      }

      if (!res.ok) {
        const text = await res.text()
        console.error(`❌ [API] ${res.status} ${url} – ${text}`)
        throw new Error(`${res.status}: ${text}`)
      }

      return res.status !== 204 ? res.json() : null
    } catch (err) {
      console.error('❌ [API] Request failed:', err)
      throw err
    }
  }

  // Convenience wrappers
  getHealth()        { return this.request('/health') }
  getReady()         { return this.request('/ready/frontend') }
  getReadyFull()     { return this.request('/ready/full') }
  getHello()         { return this.request('/hello') }

  login(credentials) {
    const body = new URLSearchParams({
      username: credentials.username,
      password: credentials.password
    })
    return this.request('/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body
    })
  }

  predictIris(payload) {
    return this.request('/iris/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  }

  predictCancer(payload) {
    return this.request('/cancer/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
  }

  trainIris(modelType = 'rf') {
    return this.request('/iris/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: modelType })
    })
  }

  trainCancer(modelType = 'bayes') {
    return this.request('/cancer/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: modelType })
    })
  }

  test401() { return this.request('/test/401') }
}

export const apiService = new ApiService();
export default apiService;
