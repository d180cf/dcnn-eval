const [, , input] = process.argv;

const fs = require('fs');
const fstext = require('./fstext');
const glob = require('glob');
const format = require('./format');

const stats = [];

let total = 0;

for (const path of glob.sync(input)) {
    const data = fstext.read(path);
    const safe = +/TS\[(.)\]/.exec(data)[1];
    const size = +/AS\[(\d+)\]/.exec(data)[1];
    const defs = +/DS\[(.)\]/.exec(data)[1];

    stats[size] = stats[size] || [0, 0, 0, 0];
    stats[size][defs + safe * 2]++; // bit 0 = who starts, bit 1 = who wins
    total++;
}

const pattern = '{0:>4} {1:>6} {2:>6} {3:>6} {4:>6}';

console.log(format(pattern, 'size', 'AA', 'DA', 'AD', 'DD'));

for (let n = 0; n < stats.length; n++) {
    if (!stats[n])
        continue;

    console.log(format(pattern, n, ...stats[n]));
}
