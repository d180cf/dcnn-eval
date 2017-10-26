
const cp = require('child_process');
const os = require('os');

const ncpus = os.cpus().length;
const processes = [];
const queue = [];

let ntasks = 0;

console.log('ncpus = ' + ncpus);

function dequeue() {
    if (!queue.length)
        return; // emit event?

    if (processes.length < ncpus - 1 && queue.length > 0) {
        const taskid = ++ntasks;
        const [command] = queue.splice(0, 1);
        const log = message => console.log('[' + process.pid + ':' + taskid + '] ' + ('' + message).trim());

        const process = cp.exec(command, (error, stdout, stderr) => {
            error && log(error);
            stderr && log(stderr);
            processes.splice(processes.indexOf(process), 1);
            dequeue();
        });

        process.stdout.on('data', data => {
            log(data + '');
        });

        log(command);
        processes.push(process);
    }
}

exports.run = function run(command) {
    queue.push(command);
    dequeue();
    // return a promise that resolves to the process object?
};


