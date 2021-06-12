sbatch_template = '''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name="{job_name}"
#
##SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,TIME_LIMIT_50
##SBATCH --mail-user={email}

#SBATCH -p main,gpu
##SBATCH -p standard
##SBATCH --gres=gpu:1

#SBATCH -t 49:59:59

#SBATCH --error="{file_err}"
#SBATCH --output="{file_out}"
##SBATCH --nodelist=ai0[1-6]

module purge
. ~/load-py.sh

# conda activate {{env_path}}

{args}
'''
