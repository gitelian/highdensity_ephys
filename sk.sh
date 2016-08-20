#!/bin/bash
# Job name:
#SBATCH --job-name=FID1152_M1
#
# Partition:
#SBATCH --partition=savio2
#SBATCH --nodes=1
#
#SBATCH --account=ac_adesnik
#SBATCH --qos=savio_normal
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
# mail alert state
#SBATCH --mail-type=all
#
# send mail to this address
#SBATCH --mail-user=gitelian@gmail.com
#
## Command(s) to run:
source activate klusta
klusta params.prm
