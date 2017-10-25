/**
 * Input: JSON file with features.
 * Output: text file with their visual representation
 */

const fs = require('fs');
const glob = require('glob');
const fspath = require('path');
const mkdirp = require('mkdirp');

const [, , input, output] = process.argv;

for (const inppath of glob.sync(input)) {
    const outpath = output.replace('*', /(\w+)\.\w+$/.exec(inppath)[1]);
    const features = JSON.parse(fs.readFileSync(inppath, 'utf8'));
    const text = features.map(m => m.map(a => a.map(x => x ? '#' : '-').join(' ')).join('\n')).join('\n\n');
    mkdirp.sync(fspath.dirname(outpath));
    fs.writeFileSync(outpath, text, 'utf8');
}
