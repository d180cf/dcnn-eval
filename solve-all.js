const fspath = require('path');
const md5 = require('md5');
const pool = require('./proc-pool');
const sgfp = require('sgf-problems');

const [, , varsDir, boardsDir] = process.argv;

for (const dir of sgfp.dirs) {
    for (const file of dir.problems) {
        const text = file + '';
        const hash = md5(text).slice(0, 7);
        const path = fspath.join(varsDir, hash + '.sgf');

        // problems marked with PL[..] and MA[..] are used by unit tests
        if (/\bPL\[\w+\]/.test(text) && /\bMA\[\w+\]/.test(text)) {
            pool.run(`node solve ${file.path} ${path} 15 7 && node vplay ${path} ${boardsDir}`);
        }
    }
}
