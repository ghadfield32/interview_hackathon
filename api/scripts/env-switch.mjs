#!/usr/bin/env node
/**
 * Environment switching utility
 * 
 * Copies environment files to api/.env for cross-platform compatibility
 * Usage: node scripts/env-switch.mjs [env-file]
 * 
 * Examples:
 *   node scripts/env-switch.mjs api/env.prod
 *   node scripts/env-switch.mjs api/env.staging
 */

import { copyFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Parse arguments - first arg after script name
const [, , src = 'env.dev'] = process.argv;

// Resolve relative to script directory (now in api/scripts/)
const srcPath = join(__dirname, '..', src);

if (!existsSync(srcPath)) {
  console.error(`❌ Environment file not found: ${srcPath}`);
  console.error('Available files:');
  console.error('  env.dev');
  console.error('  env.staging');
  console.error('  env.prod');
  process.exit(1);
}

try {
  const targetPath = join(__dirname, '..', '.env');
  copyFileSync(srcPath, targetPath);
  console.log(`✔  switched to ${src}`);
} catch (error) {
  console.error(`❌ Failed to copy environment file: ${error.message}`);
  process.exit(1);
} 
