const [, , input] = process.argv;

const fs = require('fs');
const fstext = require('./fstext');
const glob = require('glob');
const format = require('./format');

const stats = [];

let total = 0;

for (const path of glob.sync(input)) {
    const data = fstext.read(path);
    const safe = (/TS\[(\d)\]/ || []).exec(data)[1] || 2;
    const size = (/AS\[(\d+)\]/ || []).exec(data)[1] || 0;

    stats[size] = stats[size] || [0, 0, 0];
    stats[size][safe]++;
    total++;
}

const pattern = '{0:>4} {1:>6} {2:>6}';

console.log(format(pattern, 'size', 'safe', 'unsafe'));

for (let n = 0; n < stats.length; n++) {
    if (!stats[n])
        continue;

    const [safe, unsafe] = stats[n];
    console.log(format(pattern, n, safe, unsafe));
}
