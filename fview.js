/**
 * Input: JSON file with features.
 * Output: text file with their visual representation
 */

const fs = require('fs');
const fspath = require('path');
const mkdirp = require('mkdirp');
const clargs = require('command-line-args');

const args = clargs([
    { name: 'input', type: String },
    { name: 'output', type: String },
]);

try {
    const features = JSON.parse(fs.readFileSync(args.input, 'utf8'));
    const text = features.map(m => m.map(a => a.map(x => x ? '#' : '-').join(' ')).join('\n')).join('\n\n');
    mkdirp.sync(fspath.dirname(args.output));
    fs.writeFileSync(args.output, text, 'utf8');
} catch (err) {
    throw err;
}
