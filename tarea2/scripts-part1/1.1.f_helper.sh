#!/bin/bash
#PBS -l cput=150:00:00
#PBS -l walltime=150:00:00

use anaconda2
python /user/m/marvill/ANN/tarea2/scripts-part1/1.1.f.py $1
