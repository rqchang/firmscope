# ================================================================= #
# import_comp_state.R
# ================================================================= #
# Description:
# ------------
#   Pulls gvkey -> stateabbr (HQ state) mapping from WRDS comp.names
#   and saves as a flat CSV for use in Python analysis scripts.
#
# Input(s):
# ---------
#   WRDS connection
#
# Output(s):
# ----------
#   processed/Compustat/gvkey_state.csv
#     Columns: gvkey, stateabbr, loc, ipodate, dldte
#     stateabbr : 2-letter US state abbreviation (current HQ state)
#     loc       : ISO country code (use loc == 'USA' to keep US firms)
#
# Note:
# -----
#   comp.names is a static (non-time-varying) snapshot of current
#   firm headquarters.  For firms that relocated, this will reflect
#   the most recent state.  Time-varying address history is in
#   comp.addresshist if needed.
#
# ================================================================= #

rm(list = ls())

library(data.table)
library(RPostgres)

source('utils/setPaths.R')
source('utils/wrds_credentials.R')

creds <- get_wrds_credentials()
wrds  <- dbConnect(Postgres(),
                   host     = 'wrds-pgdata.wharton.upenn.edu',
                   port     = 9737,
                   user     = creds$username,
                   password = creds$password,
                   sslmode  = 'require',
                   dbname   = 'wrds')

# Pull gvkey + location fields from comp.names
state_map <- dbGetQuery(wrds,
  "SELECT gvkey, stateabbr, loc, ipodate, dldte
   FROM comp.names")

state_map <- as.data.table(state_map)
state_map <- state_map[order(gvkey)]

# Integrity check
if (any(duplicated(state_map$gvkey))) {
  stop("Duplicate gvkeys in comp.names -- unexpected.")
}

out_path <- file.path(PROCDIR, "Compustat", "gvkey_state.csv")
fwrite(state_map, out_path)
cat("Saved", nrow(state_map), "rows to", out_path, "\n")
