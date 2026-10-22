export function buildApiUrl(path, baseUrl, isDevelopment) {
  if (!path.startsWith('/') || path.startsWith('//')) {
    throw new Error('API paths must start with exactly one slash.')
  }

  const configuredBase = baseUrl?.trim().replace(/\/+$/, '')
  const apiBase = configuredBase || (isDevelopment ? 'http://127.0.0.1:8000' : '')
  if (!apiBase) {
    throw new Error('VITE_API_BASE_URL is not configured. Set it to the deployed Render API URL.')
  }

  let parsedBase
  try {
    parsedBase = new URL(apiBase)
  } catch (error) {
    throw new Error('VITE_API_BASE_URL must be an absolute HTTP(S) URL.', { cause: error })
  }
  if (!['http:', 'https:'].includes(parsedBase.protocol)) {
    throw new Error('VITE_API_BASE_URL must use HTTP or HTTPS.')
  }
  if (!isDevelopment && parsedBase.protocol !== 'https:') {
    throw new Error('VITE_API_BASE_URL must use HTTPS in production.')
  }
  return `${apiBase}${path}`
}

export function apiUrl(path) {
  const viteEnv = import.meta.env ?? {}
  return buildApiUrl(path, viteEnv.VITE_API_BASE_URL, viteEnv.DEV)
}
