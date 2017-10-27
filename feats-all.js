/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const fspath = require('path');
const glob = require('glob');
const pool = require('./proc-pool');

const [, , inputFiles, outputDir] = process.argv;

for (const path of glob.sync(inputFiles)) {
    const name = /(\w+)\.\w+$/.exec(path)[1];
    pool.run(`node feats ${path} ${fspath.join(outputDir, name)}`);
}
