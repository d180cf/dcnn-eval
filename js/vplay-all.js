/**
 * Input: SGF files with variations.
 * Output: SGF files with annotated tsumegos.
 */

const fspath = require('path');
const glob = require('glob');
const pool = require('./proc-pool');

const [, , inputFiles, outputDir] = process.argv;

for (const path of glob.sync(inputFiles)) {
    pool.run(`node js/vplay ${path} ${outputDir}`);
}
