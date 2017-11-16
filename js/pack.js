const [, , input, output] = process.argv;

const fs = require('fs');
const fstext = require('./fstext');
const glob = require('glob');

const file = fs.createWriteStream(output, { flags: 'w+' });

for (const path of glob.sync(input)) {
    const data = fstext.read(path);
    file.write(data + '\n', 'utf8');
}
