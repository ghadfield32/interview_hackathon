import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

/**
 * Vite config:
 *  - Relies on generated web/.env containing *one* VITE_API_URL derived from base keys.
 *  - Keeps normalization safeguard.
 */
export default defineConfig(({ mode }) => {
  const envDir = __dirname
  const env = loadEnv(mode, envDir, '') // loads web/.env
  const raw = env.VITE_API_URL || ''
  const trimmed = raw.replace(/\/+$/, '')
  const API_URL = /\/api\/v1$/.test(trimmed) ? trimmed : `${trimmed}/api/v1`

  if (!API_URL) {
    if (mode === 'development') {
      console.warn('[vite.config] VITE_API_URL missing – using http://127.0.0.1:8000/api/v1')
    } else {
      throw new Error('[vite.config] VITE_API_URL is required for non-dev builds')
    }
  }

  console.log('🔍 Vite Config:')
  console.log('  Mode              :', mode)
  console.log('  Loaded from       :', path.join(envDir, '.env'))
  console.log('  VITE_API_URL (raw):', raw)
  console.log('  API_URL (final)   :', API_URL)

  return {
    envDir,
    plugins: [react()],
    define: {
      __BUILD_API_URL__: JSON.stringify(API_URL),
    },
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api/v1': {
          target: 'http://127.0.0.1:8000',
            changeOrigin: true,
            secure: false,
            rewrite: p => p
        }
      }
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: false,
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom']
          }
        }
      }
    },
    esbuild: {
      logOverride: { 'this-is-undefined-in-esm': 'silent' },
      target: 'es2020',
      keepNames: true
    }
  }
})
