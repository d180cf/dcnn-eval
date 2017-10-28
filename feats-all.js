/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const fspath = require('path');
const glob = require('glob');
const pool = require('./proc-pool');

const [, , inputFiles, outputDir] = process.argv;

for (const path of glob.sync(inputFiles)) {
    const name = fspath.basename(path, fspath.extname(path)) + '.json';
    pool.run(`node feats ${path} ${fspath.join(outputDir, name)}`);
}
