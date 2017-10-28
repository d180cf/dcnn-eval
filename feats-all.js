/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const fspath = require('path');
const glob = require('glob');
const feats = require('./feats');

const [, , inputFiles, outputDir] = process.argv;

console.log('Getting the list of files...');
const paths = glob.sync(inputFiles);

paths.forEach((path, index) => {
    process.stdout.write(`${index / paths.length * 100 | 0} % = ${index} / ${paths.length}\r`);
    const name = fspath.basename(path, fspath.extname(path)) + '.json';
    feats.compute(path, fspath.join(outputDir, name)); // takes about 20 ms x 500 K files
});
