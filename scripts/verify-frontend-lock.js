#!/usr/bin/env node
/**
 * Sanity check that web/package-lock.json matches web/package.json ranges.
 * Warns when your lock is missing or stale (which breaks reproducibility with `npm ci`).
 */
import fs from 'fs';
import path from 'path';
import semver from 'semver';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');
const webDir = path.join(root, 'web');
const pkgPath = path.join(webDir, 'package.json');
const lockPath = path.join(webDir, 'package-lock.json');

function die(msg) {
  console.error(`❌ ${msg}`);
  process.exitCode = 1;
}

if (!fs.existsSync(pkgPath)) die('web/package.json not found');
if (!fs.existsSync(lockPath)) die('web/package-lock.json not found (run npm install in web/)');

const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));

const wanted = { ...pkg.dependencies, ...pkg.devDependencies };
const lockDeps = lock.packages || {};

let mismatches = 0;
for (const [name, range] of Object.entries(wanted)) {
  const lockInfo = lockDeps[`node_modules/${name}`];
  if (!lockInfo?.version) {
    console.warn(`⚠️  ${name} missing in lockfile`);
    mismatches++;
    continue;
  }
  if (!semver.satisfies(lockInfo.version, range, { includePrerelease: true })) {
    console.warn(`⚠️  ${name}@${lockInfo.version} does not satisfy ${range}`);
    mismatches++;
  }
}

if (mismatches === 0) {
  console.log('✅ Lockfile satisfies manifest ranges – suitable for npm ci reproducible installs.');
} else {
  console.log(`⚠️  ${mismatches} mismatch(es) found – run npm run frontend:rebuild-lock before CI.`);
} 
