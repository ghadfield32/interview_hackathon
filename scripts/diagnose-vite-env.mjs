#!/usr/bin/env node
/**
 * Diagnose the effective VITE_API_URL layers:
 *  1. process.env before load
 *  2. .env file content
 *  3. loadEnv outcome
 *  4. Normalized final (mirrors config-to-env logic)
 */
import fs from 'fs';
import path from 'path';

const webDir = path.join(process.cwd(), 'web');
const envFile = path.join(webDir, '.env');

function readDotEnvValue() {
  if (!fs.existsSync(envFile)) return '(no .env file)';
  const txt = fs.readFileSync(envFile,'utf8');
  const match = txt.match(/^VITE_API_URL=(.*)$/m);
  return match ? match[1] : '(not set in .env)';
}

function loadEnvFallback(mode, envDir, prefix) {
  const env = {};
  const envFile = path.join(envDir, '.env');
  if (fs.existsSync(envFile)) {
    const content = fs.readFileSync(envFile, 'utf8');
    const lines = content.split('\n');
    for (const line of lines) {
      const match = line.match(/^([^#][^=]+)=(.*)$/);
      if (match && match[1].startsWith(prefix)) {
        env[match[1]] = match[2];
      }
    }
  }
  return env;
}

const processVal = process.env.VITE_API_URL || '(unset)';
const fileVal = readDotEnvValue();
const loaded = loadEnvFallback('development', webDir, '');
const loadEnvVal = loaded.VITE_API_URL || '(unset)';

function normalizeApi(url){
  if (!url || url.startsWith('(')) return url;
  let base = url.replace(/\/+$/,'');
  if (!/\/api\/v1$/.test(base)) base += '/api/v1';
  return base;
}
const final = normalizeApi(processVal !== '(unset)' ? processVal : loadEnvVal);

console.log('--- VITE_API_URL DIAG ---');
console.log('process.env       :', processVal);
console.log('.env file         :', fileVal);
console.log('loadEnv() result  :', loadEnvVal);
console.log('Final normalized  :', final);
console.log('Provenance        :', fs.existsSync(path.join(webDir,'__vite_api_url.meta.json'))
  ? JSON.parse(fs.readFileSync(path.join(webDir,'__vite_api_url.meta.json'),'utf8'))
  : '(no meta)'); 