#!/bin/bash
#PBS -l cput=150:00:00
#PBS -l walltime=150:00:00

/user/m/marvill/anaconda2/bin/python2 /user/m/marvill/ANN/tarea2/scripts_part2/mlp_semi_DAE.py $1
