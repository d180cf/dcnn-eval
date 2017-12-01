/**
 * Input: SGF files from the `sgf-problems` module.
 * Output: SGF files with variations.
 */

const fspath = require('path');
const md5 = require('md5');
const pool = require('./proc-pool');
const sgfp = require('sgf-problems');

const [, , outputDir, maxTreeSize, minAreaSize, branchingFactor] = process.argv;

for (const dir of sgfp.dirs) {
    for (const file of dir.problems) {
        const ext = fspath.extname(file.path);
        const name = fspath.basename(file.path).slice(0, -ext.length);
        const text = file + '';
        const hash = md5(text).slice(0, 7);
        const path = fspath.join(outputDir, name, hash + '.sgf');

        // problems marked with PL[..] and MA[..] are used by unit tests
        if (/\bPL\[\w+\]/.test(text) && /\bMA\[\w+\]/.test(text))
            pool.run(`node js/solve ${file.path} ${path} ${maxTreeSize} ${minAreaSize} ${branchingFactor}`);
    }
}
