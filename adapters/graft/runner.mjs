// Trusted entrypoint. Never imports or executes target repository modules.
import {extract, PARSER} from './extract.mjs';
import {createHash} from 'node:crypto';
const SCHEMA = 'jitmind.graft-parser/v1';
let bytes = 0;
const chunks = [];
for await (const chunk of process.stdin) {
  bytes += chunk.length;
  if (bytes > 8 * 1024 * 1024) throw Error('input_budget');
  chunks.push(chunk);
}
const request = JSON.parse(Buffer.concat(chunks).toString('utf8'));
if (request.schema !== SCHEMA || !/^[a-f0-9]{64}$/.test(request.snapshot_id) ||
    !Array.isArray(request.files) || request.files.length > 128) throw Error('protocol');
let total = 0;
const files = request.files.map(file => {
  if (typeof file.source !== 'string' || typeof file.path !== 'string' ||
      !/^[a-f0-9]{64}$/.test(file.digest)) throw Error('protocol');
  const size = Buffer.byteLength(file.source);
  total += size;
  if (size > 131072 || total > 2097152 ||
      createHash('sha256').update(file.source).digest('hex') !== file.digest) throw Error('input_budget');
  const result = /\.pyi?$/.test(file.path) ? extract(file.source) : {status: 'unsupported', symbols: []};
  return {path: file.path, digest: file.digest, ...result};
});
const output = JSON.stringify({schema: SCHEMA, snapshot_id: request.snapshot_id, parser: PARSER, files});
if (Buffer.byteLength(output) > 262144) throw Error('output_budget');
process.stdout.write(output);
