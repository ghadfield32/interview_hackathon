#!/usr/bin/env node
/**
 * Revised generator:
 *  - Introduces separation of LOCAL/STAGING/RAILWAY base keys (not auto-exposed).
 *  - Derives exactly one VITE_API_URL based on APP_ENV.
 *  - Still copies config.yaml into web/ for transparency.
 *  - Ensures old config.yaml is deleted before copying fresh one.
 *  - Adds APP_ENV to the .env file for frontend awareness.
 *
 * Selection rules:
 *   prod|production -> RAILWAY_VITE_API_BASE
 *   staging         -> STAGING_VITE_API_BASE || LOCAL_VITE_API_BASE
 *   default (dev)   -> LOCAL_VITE_API_BASE
 *
 * Only the final derived VITE_API_URL is written to web/.env (plus any other VITE_* keys that already exist).
 *
 * Safe: Non-VITE_* keys stay private (not shipped to client bundle).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as yaml from 'yaml';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const root       = path.resolve(__dirname, '..');
const webDir     = path.join(root, 'web');

function log(...args)  { console.log('[config-to-env]', ...args); }
function fail(msg)     { console.error('❌ [config-to-env]', msg); process.exit(1); }

const cliEnvArg = process.argv.slice(2).find(a => !a.startsWith('-'));
const targetEnv = (process.env.APP_ENV || cliEnvArg || 'dev').toLowerCase();

const cfgPath = path.join(root, 'config.yaml');
if (!fs.existsSync(cfgPath)) fail(`config.yaml not found at ${cfgPath}`);

let doc;
try {
  doc = yaml.parse(fs.readFileSync(cfgPath, 'utf8'));
} catch (e) {
  fail(`YAML parse error: ${e.message}`);
}

if (!doc.default) fail('Missing "default" section in config.yaml');

const merged = { ...doc.default, ...(doc[targetEnv] || {}) };

// ---- Derive effective base -------------------------------------------------
const localBase    = merged.LOCAL_VITE_API_BASE;
const stagingBase  = merged.STAGING_VITE_API_BASE || localBase;
const railwayBase  = merged.RAILWAY_VITE_API_BASE;

let effectiveBase;
if (['prod','production'].includes(targetEnv)) {
  effectiveBase = railwayBase;
} else if (targetEnv === 'staging') {
  effectiveBase = stagingBase;
} else {
  effectiveBase = localBase;
}

if (!effectiveBase) {
  fail(`No effective API base resolved (check LOCAL_VITE_API_BASE / RAILWAY_VITE_API_BASE keys).`);
}

// Normalize and append /api/v1 if needed
function normalizeApiBase(b) {
  let base = b.trim().replace(/\/+$/, '');
  if (!/\/api\/v1$/.test(base)) base += '/api/v1';
  return base;
}
const finalApiUrl = normalizeApiBase(effectiveBase);

// Collect any *existing* merged VITE_ keys (other than API URL we now control)
const viteEntries = Object.entries(merged)
  .filter(([k]) => k.startsWith('VITE_') && k !== 'VITE_API_URL')
  .map(([k, v]) => [k, v]);

// Inject our derived VITE_API_URL at the top for visibility
viteEntries.unshift(['VITE_API_URL', finalApiUrl]);

// Compose .env content
const lines = viteEntries.map(([k, v]) => `${k}=${v}`);

// Add APP_ENV to the .env file for frontend awareness
lines.push(`APP_ENV=${targetEnv}`);

const envOut = lines.join('\n') + '\n';

if (!fs.existsSync(webDir)) fail(`web directory not found at ${webDir}`);

// Delete old config.yaml from web/ if it exists
const webConfigPath = path.join(webDir, 'config.yaml');
if (fs.existsSync(webConfigPath)) {
  try {
    fs.unlinkSync(webConfigPath);
    log('Deleted old web/config.yaml');
  } catch (e) {
    log('Warning: Could not delete old web/config.yaml:', e.message);
  }
}

const envPath = path.join(webDir, '.env');
fs.writeFileSync(envPath, envOut, 'utf8');

// Copy fresh config.yaml for inspection
fs.copyFileSync(cfgPath, webConfigPath);

log(`Environment         : ${targetEnv}`);
log(`Resolved API base   : ${effectiveBase}`);
log(`VITE_API_URL (final): ${finalApiUrl}`);
log(`VITE keys written   : ${viteEntries.length}`);
log(`APP_ENV set to      : ${targetEnv}`);
log(`Config copied to    : ${webConfigPath}`); 

