/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const [, script, inputDir, outputDir] = process.argv;

const fspath = require('path');
const glob = require('glob');
const feats = require('./feats');

console.log('Getting the list of files...');
const paths = glob.sync(fspath.join(inputDir, '**', '*.sgf'));
console.log(paths.length + ' files total');

console.log('Computing features...');
for (const path of paths) {
    const relpath = fspath.relative(inputDir, path);
    const output = fspath.join(outputDir, relpath).replace('.sgf', '.json');

    feats.compute(path, output); // takes about 20 ms x 500 K files
}
