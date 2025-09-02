#!/usr/bin/env node
/**
 * Copy root config.yaml → api/config.yaml
 * Usage: node api/scripts/config-sync.mjs
 */

import { copyFileSync, existsSync } from 'fs';
import { fileURLToPath }      from 'url';
import { dirname, join }      from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);

const rootYaml = join(__dirname, '..', '..', 'config.yaml');
const destYaml = join(__dirname, '..', 'config.yaml');

if (!existsSync(rootYaml)) {
  console.error(`❌  config.yaml not found at ${rootYaml}`);
  process.exit(1);
}

try {
  copyFileSync(rootYaml, destYaml);
  console.log(`✔  overwritten api/config.yaml from root config.yaml`);
} catch (error) {
  console.error(`❌  Failed to copy config: ${error.message}`);
  process.exit(1);
} 
