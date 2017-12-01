/**
 * Input: SGF files with variations.
 * Output: SGF files with annotated tsumegos.
 */

const fs = require('fs');
const fspath = require('path');
const glob = require('glob');
const pool = require('./proc-pool');

const [, , inputFiles, outputDir] = process.argv;

for (const path of glob.sync(inputFiles)) {
    const stat = fs.lstatSync(path);

    if (!stat.isFile())
        continue;

    pool.run(`node js/vplay ${path} ${outputDir}`);
}
