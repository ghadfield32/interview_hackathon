// scripts/sync-env.js
import { copyFileSync } from 'fs';
import { resolve } from 'path';

console.log('🔄 Syncing environment file...');

const from = resolve(process.cwd(), 'web', 'env.template');
const to   = resolve(process.cwd(), 'web', '.env');

try {
  copyFileSync(from, to);
  console.log(`✅ Copied ${from} → ${to}`);
} catch (e) {
  console.error(`❌ Failed to copy env.template:`, e.message);
  process.exit(1);
} 
