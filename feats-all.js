/**
 * Input: SGF files with tsumegos.
 * Output: JSON files with features.
 */

const [, script, inputFiles, outputDir, prefix] = process.argv;

if (!prefix) {
    const pool = require('./proc-pool');

    for (let i = 0; i < 16; i++)
        pool.run(`node ${script} ${inputFiles} ${outputDir} ${i.toString(16)}`);
} else {
    const fs = require('fs');
    const fspath = require('path');
    const glob = require('glob');
    const feats = require('./feats');
        
    console.log('Getting the list of files...');
    const paths = glob.sync(inputFiles);

    console.log('Computing features...');
    paths.forEach((path, index) => {
        const name = fspath.basename(path, fspath.extname(path)) + '.json';
        const output = fspath.join(outputDir, name);

        if (prefix && !name.startsWith(prefix))
            return;

        if (!fs.existsSync(output))
            feats.compute(path, output); // takes about 20 ms x 500 K files
    });
}
