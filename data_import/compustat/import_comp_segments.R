# ================================================================= #
# import_comp_segments.R ####
# ================================================================= #
# Description:
# ------------
#   Downloads Compustat business segment annual data from WRDS.
#   Used to construct segment-based firm scope measures:
#     n_segments  = number of distinct business segments
#     seg_hhi     = Herfindahl index of segment revenue shares
#     seg_diversity = 1 - seg_hhi  (Berry entropy diversification)
#
# Input(s):
# ---------
#   WRDS connection (comp.seg_ann)
#
# Output(s):
# ----------
#   data/raw/Compustat/compustat_segments.rds
#
# Date:
# ----------
#   2026-03-30
#
# Author(s):
# ----------
#   Ruiquan Chang, chang.2590@osu.edu
#
# Compustat Segment Variables:
# * GVKEY   = Global Company Key
# * DATADATE= Data Date
# * SID     = Segment Identifier
# * STYPE   = Segment Type (BUSSEG / GEOSEG / OPSEG)
# * SNMS    = Segment Name
# * SALES   = Segment Net Sales/Revenues
# * IBC     = Segment Income Before Extraordinary Items
# * DP      = Segment Depreciation
# * CAPX    = Segment Capital Expenditures
# * AT      = Segment Identifiable Assets
#
# ================================================================= #


# ================================================================= #
# Environment ####
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


# ================================================================= #
# Read data ####
# ================================================================= #
print('Downloading Compustat segment data from WRDS (comp.seg_ann)...')

varlist <- c('gvkey', 'datadate', 'sid', 'stype', 'snms',
             'sales', 'ibc', 'dp', 'capx', 'at')

sql <- sprintf("SELECT %s FROM comp.seg_ann
                WHERE stype = 'BUSSEG'",
               paste(varlist, collapse = ','))

seg <- dbGetQuery(wrds, sql)


# ================================================================= #
# Clean and organize ####
# ================================================================= #
seg <- as.data.table(seg)
seg <- seg[order(gvkey, datadate, sid)]

# Primary key: gvkey + datadate + sid
check <- nrow(seg[is.na(gvkey) | is.na(datadate) | is.na(sid)]) == 0
if (!check) print('WARNING: Missing gvkey/datadate/sid.')

if (any(duplicated(seg, by = c('gvkey', 'datadate', 'sid')))) {
  print('WARNING: Duplicates on (gvkey, datadate, sid) -- keeping last.')
  seg <- unique(seg, by = c('gvkey', 'datadate', 'sid'), fromLast = TRUE)
}

setkeyv(seg, c('gvkey', 'datadate', 'sid'))


# ================================================================= #
# Write data ####
# ================================================================= #
saveRDS(seg, paste0(RAWDIR, 'Compustat/compustat_segments.rds'))
print(sprintf('Segment data saved: %d rows, %d gvkeys, years %d-%d.',
              nrow(seg),
              seg[, uniqueN(gvkey)],
              seg[, min(as.integer(format(as.Date(datadate), '%Y')), na.rm = TRUE)],
              seg[, max(as.integer(format(as.Date(datadate), '%Y')), na.rm = TRUE)]))
