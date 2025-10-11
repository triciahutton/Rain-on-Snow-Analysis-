#!/bin/sh

#SBATCH --partition=t1standard
#SBATCH --account=dyndown
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=phutton5@alaska.edu
#SBATCH --mail-type=FAIL
#SBATCH --time=10:00:00
#SBATCH --output=output_ROS_Precip_log.%j

ulimit -s unlimited
ulimit -l unlimited

source $(conda info --base)/etc/profile.d/conda.sh

conda activate dyndown
echo "Conda environment activated"

chmod +x ROS_Precip_by_season.py  

echo "starting ROS with temp pressure levels   " 

python -u ROS_Precip_by_season.py --start-year 1984 --n-years 1 > log_ROS_Precip_out.${SLURM_JOB_ID} 2>&1


echo "script complete!"
#echo "This has been done using Python, next test is with SRUN using MPI"

