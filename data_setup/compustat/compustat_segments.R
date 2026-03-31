# ================================================================= #
# compustat_segments.R ####
# ================================================================= #
# Description:
# ------------
#   Aggregates Compustat segment-level data to firm-year panel and
#   constructs segment-based firm scope measures.
#
# Input(s):
# ---------
#   data/raw/Compustat/compustat_segments.rds
#
# Output(s):
# ----------
#   data/processed/Compustat/compustat_segments_annual.csv
#
#   Variables:
#     gvkey          Global Company Key
#     fyear          Fiscal year (from datadate year)
#     n_segments     Number of reported business segments
#     seg_sales_tot  Total segment revenue ($M)
#     seg_hhi        Herfindahl index of revenue shares (0=monopoly, 1=fully diversified)
#                    seg_hhi = sum_k (s_k)^2  where s_k = seg_k_sales / total_sales
#     seg_diversity  Berry diversification index = 1 - seg_hhi
#                    = 0 for single-segment firms; approaches 1 for many equal segments
#
# Note on single-segment firms:
#   Compustat reports a single BUSSEG row for undiversified firms.
#   For these, n_segments = 1, seg_hhi = 1, seg_diversity = 0.
#
# Date:
# ----------
#   2026-03-30
#
# Author(s):
# ----------
#   Ruiquan Chang, chang.2590@osu.edu
#
# ================================================================= #


# ================================================================= #
# Environment ####
# ================================================================= #
rm(list = ls())

library(data.table)

source('utils/setPaths.R')


# ================================================================= #
# Read data ####
# ================================================================= #
seg <- readRDS(paste0(RAWDIR, 'Compustat/compustat_segments.rds'))

# Extract fiscal year from datadate
seg[, fyear := as.integer(format(as.Date(datadate), '%Y'))]

# Drop rows with missing or non-positive segment sales
# (keep zero-sales segments only for counting; exclude NA from HHI calc)
seg_count <- seg[!is.na(sales) | !is.na(sid)]   # all rows for counting
seg_hhi   <- seg[!is.na(sales) & sales > 0]      # only positive-sales rows for HHI


# ================================================================= #
# Construct firm-year measures ####
# ================================================================= #

# --- 1. n_segments: count distinct SIDs per firm-year ---
n_seg <- seg_count[, .(n_segments = uniqueN(sid)), by = .(gvkey, fyear)]

# --- 2. Segment HHI from revenue shares ---
seg_hhi[, seg_sales_tot := sum(sales, na.rm = TRUE), by = .(gvkey, fyear)]
seg_hhi[seg_sales_tot > 0, share := sales / seg_sales_tot]
hhi   <- seg_hhi[, .(seg_hhi      = sum(share^2,   na.rm = TRUE),
                     seg_sales_tot = first(seg_sales_tot)),
                 by = .(gvkey, fyear)]

# --- 3. Merge and derive diversity index ---
firm_yr <- merge(n_seg, hhi, by = c('gvkey', 'fyear'), all.x = TRUE)
firm_yr[, seg_diversity := 1 - seg_hhi]

# For firms with n_segments == 1 and missing HHI (no positive sales reported),
# impute seg_hhi = 1, seg_diversity = 0
firm_yr[n_segments == 1 & is.na(seg_hhi), ':='(seg_hhi = 1, seg_diversity = 0)]

firm_yr <- firm_yr[order(gvkey, fyear)]


# ================================================================= #
# Check ####
# ================================================================= #
print(sprintf('Segment panel: %d firm-year obs, %d unique gvkeys, years %d-%d.',
              nrow(firm_yr),
              firm_yr[, uniqueN(gvkey)],
              min(firm_yr$fyear, na.rm = TRUE),
              max(firm_yr$fyear, na.rm = TRUE)))

print('Distribution of n_segments:')
print(firm_yr[, .N, by = n_segments][order(n_segments)][1:15])

print('Summary of seg_diversity:')
print(summary(firm_yr$seg_diversity))


# ================================================================= #
# Write ####
# ================================================================= #
fwrite(firm_yr, paste0(PROCDIR, 'Compustat/compustat_segments_annual.csv'))
print('Segment firm-year panel saved.')
