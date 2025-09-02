#!/usr/bin/env node
/**
 * Quick dependency + audit snapshot for the frontend.
 * Prints versions of key packages and shows who depends on deprecated modules.
 */
import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');
const webDir = path.join(root, 'web');

function run(cmd) {
  console.log(`\n$ ${cmd}`);
  try {
    const out = execSync(cmd, { stdio: 'pipe', cwd: webDir, encoding: 'utf8', maxBuffer: 10_000_000 });
    console.log(out.trim());
  } catch (err) {
    console.error(err.stdout?.toString() || err.message);
  }
}

console.log('🔍 Frontend dependency diagnostics');
run('node -v');
run('npm -v');
run('npm ls --depth=2 inflight || true');
run('npm ls --depth=2 glob || true');
run('npm ls --depth=2 rimraf || true');
run('npm ls --depth=1 eslint || true');
run('npm audit --omit=dev || true');  // prod issues
run('npm audit || true');             // full tree
console.log('\nDone.'); 
