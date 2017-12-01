const [, , input] = process.argv;

const fs = require('fs');
const fstext = require('./fstext');
const glob = require('glob');
const format = require('./format');

const stats = [];

let total = 0;

for (const path of glob.sync(input)) {
    const [, _size, _safe] = /\/(\d+)\/([+-=])\//.exec(path);

    const size = +_size;
    const safe = { '+': 0, '-': 1, '=': 2 }[_safe];

    stats[size] = stats[size] || [0, 0, 0];
    stats[size][safe]++;
    total++;
}

const pattern = '{0:>4} {1:>6} {2:>6} {3:>6}';

console.log(format(pattern, 'size', 'safe', 'dead', ''));

for (let n = 0; n < stats.length; n++) {
    if (!stats[n])
        continue;

    console.log(format(pattern, n, ...stats[n]));
}
