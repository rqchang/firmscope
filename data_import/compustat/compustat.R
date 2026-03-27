# ================================================================= #
# crsp.R ####
# ================================================================= #
# Description:
# ------------
#     Runs all Compustat data pulls and initial cleaning
#
# Input(s):
# ---------
#     None
#
# Output(s):
# ----------
#     None
#
# Author(s):
# ----------
#     Ruiquan Chang, rqchang@mit.edu
#
# Additional note(s):
# ----------
#     
# ================================================================= #

# ================================================================= #
# Environment ####
# ================================================================= #

# Clear workspace
rm(list = ls())

# ================================================================= #
# Source scripts ####
# ================================================================= #

source("./data/compustat/import_comp_funda.R")
source("./data/compustat/import_comp_fundq.R")
source("./data/compustat/import_comp_auxiliaries.R")
