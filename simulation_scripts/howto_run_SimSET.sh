#! /bin/sh

# This is an example run with the templates distributed with STIR (appropriate for the HR+).
# The code below works in bash, sh, ksh etc, but needs to be modified for csh.
# Authors: Kris Thielemans
#
#
#  Copyright (C) 2005 - 2006, Hammersmith Imanet Ltd
#  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
#  This file is part of STIR.
#
#  SPDX-License-Identifier: Apache-2.0
#
#  See STIR/LICENSE.txt for details

# adjust location of SimSET to where you have it installed.
SIMSET_DIR=/home/kathy/2.9.2
export SIMSET_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR is: $SCRIPT_DIR"
#All STIR utilities/scripts have to be in your path, e.g. if your 
#INSTALL_PREFIX was ~/STIR-bin, you could do
#   PATH=$PATH:~/STIR-bin/bin

# generate emission image
# use different sizes for the attenuation and activitiy
generate_image generate_activity_cylinder.par
generate_image generate_attenuation_cylinder.par

# give the simulation a name. All output files will go into a new subdirectory of this name
PROJECT=/mnt/e/PET_project/PET_simulations_new
DIR_OUTPUT=$PROJECT/simulations/sim_test_10_en_bin__1B_decays_3true_max_ring_diff_11_resolution_10

# number of decays to simulate
PHOTONS=1000000000
# specify names/locations of input files
EMISS_DATA=activity_cylinder.hv
ATTEN_DATA=attenuation_cylinder.hv
templ_dir=`pwd`
TEMPLATE_PHG=${templ_dir}/template_phg.rec
TEMPLATE_BIN=${templ_dir}/template_bin.rec
TEMPLATE_DET=${templ_dir}/template_det.rec
# specify scanner
SCANNER="ECAT HR+"
# maximum ring difference to store in conversion from SimSET to Interfile projdata
NUM_SEG=11
# export all variables
export DIR_OUTPUT EMISS_DATA ATTEN_DATA TEMPLATE_PHG TEMPLATE_BIN TEMPLATE_DET 
export PHOTONS NUM_SEG SCANNER

# set the simulation going
echo "Running local SimSET pipeline..."
"$SCRIPT_DIR/run_SimSET_local.sh"
