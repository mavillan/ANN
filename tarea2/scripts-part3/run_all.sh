#!/bin/bash

qsub -q gpuk d.sh
qsub -q gpuk g.sh
qsub -q gpuk i.sh
qsub -q gpuk 3-2.sh
qsub -q gpuk 3-4.sh
qsub -q gpuk 5-2.sh
qsub -q gpuk 5-4.sh
qsub -q gpuk 7-2.sh
qsub -q gpuk 7-4.sh
qsub -q gpuk 9-2.sh
qsub -q gpuk 9-4.sh
qsub -q gpuk 16-3-2.sh
qsub -q gpuk 32-3-2.sh
qsub -q gpuk 64-3-2.sh
qsub -q gpuk extra.sh