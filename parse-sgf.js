const sgf = require('sgf-problems');

for (const dir of sgf.dirs)
    for (const sgf of dir.problems)
        console.log(sgf.path + ':\n' + sgf + '\n');
