/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const fs = require('fs');
const fspath = require('path');
const glob = require('glob');
const feats = require('./feats');

const [, , inputFiles, outputDir] = process.argv;

console.log('Getting the list of files...');
const paths = glob.sync(inputFiles);

paths.forEach((path, index) => {
    process.stdout.write(`${index / paths.length * 100 | 0} % = ${index} / ${paths.length}\r`);
    
    const name = fspath.basename(path, fspath.extname(path)) + '.json';
    const output = fspath.join(outputDir, name);

    if (!fs.existsSync(output))
        feats.compute(path, output); // takes about 20 ms x 500 K files
});
