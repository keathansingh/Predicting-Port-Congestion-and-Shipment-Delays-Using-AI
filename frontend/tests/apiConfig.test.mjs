import assert from 'node:assert/strict'
import test from 'node:test'

import { buildApiUrl } from '../src/apiConfig.mjs'

test('API URL uses the configured HTTPS backend and joins paths once', () => {
  assert.equal(
    buildApiUrl('/api/v1/summary', 'https://port-api.onrender.com/', false),
    'https://port-api.onrender.com/api/v1/summary',
  )
})

test('API URL uses the local backend only during development', () => {
  assert.equal(
    buildApiUrl('/health', undefined, true),
    'http://127.0.0.1:8000/health',
  )
})

test('production requires a configured HTTPS backend', () => {
  assert.throws(
    () => buildApiUrl('/api/v1/summary', undefined, false),
    /VITE_API_BASE_URL is not configured/,
  )
  assert.throws(
    () => buildApiUrl('/api/v1/summary', 'http://127.0.0.1:8000', false),
    /must use HTTPS in production/,
  )
})

test('API URL rejects malformed paths and non-HTTP URLs', () => {
  assert.throws(
    () => buildApiUrl('//other-host/api', 'https://port-api.onrender.com', false),
    /exactly one slash/,
  )
  assert.throws(
    () => buildApiUrl('/health', 'file:///tmp/api', true),
    /must use HTTP or HTTPS/,
  )
})
