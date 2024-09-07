#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --mail-user=facundoy@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#SBATCH --time=14:00:00
#SBATCH --partition=alrodri-a100
#SBATCH --output=/home/%u/%x-%j.log

# The application(s) to execute along with its input arguments and options:

python main.py
python graph.py