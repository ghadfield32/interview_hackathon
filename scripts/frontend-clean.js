#!/usr/bin/env node
/**
 * Clean the frontend install in web/ in a *cross-platform* way.
 * - Deletes node_modules
 * - Optionally deletes package-lock.json when --zap-lock passed
 * - Never follows symlinks outside repo
 *
 * Use via: npm run frontend:clean [-- --zap-lock]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');
const webDir = path.join(root, 'web');

const args = process.argv.slice(2);
const zapLock = args.includes('--zap-lock');

function safeRemove(targetPath) {
  if (!fs.existsSync(targetPath)) return false;
  try {
    // fs.rmSync is cross-platform, supports recursive+force (Node 14+)
    fs.rmSync(targetPath, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
    console.log(`Removed: ${path.relative(root, targetPath)}`);
    return true;
  } catch (err) {
    console.error(`Failed to remove ${targetPath}:`, err);
    process.exitCode = 1;
    return false;
  }
}

console.log('🧹 Frontend clean start');
safeRemove(path.join(webDir, 'node_modules'));
if (zapLock) {
  safeRemove(path.join(webDir, 'package-lock.json'));
}
console.log('🧹 Frontend clean complete'); 
