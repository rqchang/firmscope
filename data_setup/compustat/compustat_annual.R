# ================================================================= #
# compustat_annual.R ####
# ================================================================= #
# Description:
# ------------
#   Create compustat annual (processed data) data set.
#   It includes a bunch of variables like me, be, leverage ratios, ind concentration, etc.
#
# Input(s):
# ---------
#   1. Compustat annual raw data:
#     data/raw/compustat/compustat_funda.rds
#     Data/raw/compustat/compustat_names.rds
#
#   2. CRSP names and link raw data:
#     data/raw/CRSP/stocknames.rds
#     data/raw/CRSP/crspq_ccmxpf_lnkhist.rds
#
# Output(s):
# ----------
#   Compustat annual processed data:
#     data/processed/compustat/compustat_annual.rds
#
# Date:
# ----------
#   2023-02-07
#   update: 2026-01-22
#
# Author(s):
# ----------
#   Lira Mota Mertens, liramota@mit.edu
#   Ruiquan Chang, chang.2590@osu.edu
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
print('Started script to compustat annual.')

# Import libraries
library(data.table) 
library(zoo)
library(lubridate)
library(ggplot2)

# Source helper scripts
source('utils/setPaths.R')
source('utils/wrds_credentials.R')
source('utils/f_compustat_variables_calculations.R')


# ================================================================= #
# Read data ####
# ================================================================= #
# Raw data
adata <- readRDS(paste0(RAWDIR, "Compustat/compustat_funda.rds")) # comp annual
names <- readRDS(paste0(RAWDIR, "Compustat/compustat_names.rds")) # names table - get latest SIC
loc <- readRDS(paste0(RAWDIR, "Compustat/compustat_conml.rds")) # state variable
stock_names <- readRDS(paste0(RAWDIR, "CRSP/stocknames.rds")) # names table - get latest SIC
link <- readRDS(paste0(RAWDIR, "CRSP/crspq_ccmxpf_lnkhist.rds")) # link CRSP and Compustat

# Check data integrity
# Drop fyear duplicates
# gvkey and fyear duplicates happen when a firms changes its fiscal year end
print(sprintf('Rows dropped because of missing fyear: %.2f%%', 
              (adata[,mean(is.na(fyear))]*100)))
adata <- adata[!is.na(fyear)]
  
print(sprintf('Rows dropped because of duplicated (datadate, gvkey): %.0f', 
              (sum(duplicated(adata,by=c('gvkey', 'datadate'))))))
print(sprintf('Rows dropped because of duplicated (fyear, gvkey): %.0f', 
              (sum(duplicated(adata,by=c('gvkey', 'fyear'))))))

adata <- adata[order(gvkey, datadate, fyear)]
adata <- unique(adata, by = c('gvkey', 'fyear'), fromLast = TRUE)


# ================================================================= #
# Merge names table ####
# ================================================================= #
adata <- merge(names[,.(gvkey, sic, naics)], adata, by = c('gvkey'), all.y = TRUE)
adata <- merge(adata, loc[, .(gvkey, state, city, county)], by = c('gvkey'), all.x = TRUE)

# Guarantee the right ordering
adata <- adata[order(gvkey, fyear)]


# ================================================================= #
# Fill gaps of missing data ####
# ================================================================= #
# Important to calculate lag variables
adata[, sdate := min(fyear, na.rm = TRUE), by = gvkey]
adata[, edate := max(fyear, na.rm = TRUE), by = gvkey]

cdata <- melt.data.table(dcast.data.table(adata, gvkey~fyear, value.var = 'datadate'),
                         id.vars = 'gvkey', variable.name = 'fyear', 
                         value.name = 'datadate', variable.factor = FALSE)
cdata[, fyear:=as.numeric(fyear)]
cdata <- merge(cdata[, .(gvkey, fyear)], 
               unique(adata[, .(gvkey, sdate, edate)]), 
               by = c('gvkey'))
cdata <- cdata[fyear>=sdate & fyear<=edate]

adata <- merge(cdata[, .(gvkey, fyear)], adata, by = c('gvkey', 'fyear'), all.x = TRUE)
rm(cdata)
adata[, c('sdate','edate') := NULL]


# ================================================================= #
# Create annual variables of interest ####
# ================================================================= #
# -------------------------------------------------------------#
# Market capitalization 
adata[, me_comp := (prcc_f * csho)]
# Calculate BE 
adata[,'be' := calculate_be(adata, freq='annual')]

# -------------------------------------------------------------#
# Calculate Operating Profits
adata[,'costt' := sum(cogs, xsga, xint, na.rm = TRUE), by = 1:NROW(adata)]
adata[is.na(cogs) & is.na(xsga) & is.na(xint), costt := NA]
summary(adata$costt) #TODO: what is negative cost?
adata[,'op' := sale - costt]

# -------------------------------------------------------------#
# From Yueran Ma (2018) 
# What is statetement of cash flows? prstkcc is cash flow statement, but it is available only for 2% of the data.
# Note - DECISION - substitute missing for zero
# Net Equity Repurchase (Millions)
adata[,"ner":=sum(prstkc, -sstk, na.rm = TRUE), by = 1:NROW(adata)]

print(sprintf('We calculated Net Equity Repurchase (NER). 
         We replace %.2f percent of the data to zero due missing values.', 
        (adata[, mean(is.na(prstkc) & is.na(sstk))] * 100)))

# Net Equity Repurchase + Dividend (Millions)
# Note: we are using total dividends 
adata[, "nerd" := sum(dvt, prstkc, -sstk, na.rm = TRUE),by=1:NROW(adata)]
print(sprintf('We calculated total payment to shareholders (NERT=NER+DIV). 
         We replace %.2f percent of the data to zero due missing values.', 
        (adata[, mean(is.na(dv) & is.na(prstkc) & is.na(sstk))] * 100)))

# Net Debt Issuance (Millions)
adata[,"ndi" := sum(dltis, -dltr, na.rm = TRUE), by = 1:NROW(adata)]
print(sprintf('We calculated Net Debt Issuance (NDI). 
         We replace %.2f percent of the data to zero due missing values.', 
        (adata[, mean(is.na(dltis) & is.na(dltr))] * 100)))

# -------------------------------------------------------------#
# External Cash-Flows Variales (Eisfeldt and Muir (JME, 2016)) 
# Flows to equity
## - Sale of common and pref. stocks + purchase of common and pref. stocks + dividends
adata [, cfe := sum(-sstk, prstkc, dv, na.rm = TRUE), by = 1:NROW(adata)]

# Flows to debt : two first terms llong term debt change.
## - long term debt issuance + long term debt reduction + changes in current debt + interest paid
adata[, cfd := sum(-dltis, dltr, dlcch, xint, na.rm=TRUE), by = 1:NROW(adata)] #na.fill(dlcch,0)

# Flows to external fiance
adata[, cfef := (cfe + cfd)] # EM take negative of this value

# Check missing
adata[, mean(is.na(cfe))]
adata[, mean(is.na(cfd))]

# Missing Data as of now is consider zero!
adata[, mean(is.na(sstk)&is.na(prstkc)&is.na(dv))] #na.fill(dlcch,0)
adata[, mean(is.na(dltis)&is.na(dltr)&is.na(xint))] #na.fill(dlcch,0)

# Notice that dle is mostly missing due to dlcch -  I don't understand where those "ch" varaibles are coming from
adata[, mean(is.na(dltis))]
adata[, mean(is.na(dltr))]
adata[, mean(is.na(dlcch))]# Makes sense, since companies don't change in fact
adata[, mean(is.na(xint))]

# Normalize by BE or AT
colvars <- c('che', 'ch', 'cfe', 'cfd', 'cfef')
adata[be > 0, (paste0(colvars,'be')) := lapply(.SD, function(x){x/be}), .SDcols = colvars]
adata[at > 0, (paste0(colvars,'at')) := lapply(.SD, function(x){x/at}), .SDcols = colvars]

# Normalize by lag BE or AT
adata <- adata[order(gvkey, fyear)]
adata[, atlag := shift(at), by = gvkey]
adata[, belag := shift(be), by = gvkey]

adata[belag > 0, (paste0(colvars,'belag')) := lapply(.SD, function(x){x/belag}), .SDcols = colvars]
adata[atlag > 0, (paste0(colvars,'atlag')) := lapply(.SD, function(x){x/atlag}), .SDcols = colvars]

# Look at GM data
# adata[gvkey=='005073', .(gvkey, conm, fyear, che, cfe, cfd)][order(fyear)]

# -------------------------------------------------------------#
# Total Debt (Change)
adata[, 'tdebt' := sum(dltt, dlc, na.rm = TRUE),by = 1:NROW(adata)]
# missing total debt is considered to be zero
adata[, 'nc_tdebtq' := tdebt-shift(tdebt), by = .(gvkey)]
# Total Long Term Debt (Change)
adata[, 'ldebt' := sum(dltt, na.rm=TRUE), by = 1:NROW(adata)]
adata[, 'nc_ldebt' := ldebt-shift(ldebt), by = .(gvkey)]

# -------------------------------------------------------------#
# Leverage
# http://onlinelibrary.wiley.com/doi/10.1111/j.1468-2443.2010.01125.x/epdf
# FDAT: financial-debt-to-assets ratio: FD/AT = (DLTT+DLC)/AT: Welch argues it is 
#       a wrong measure to analyze capital structure
# LTAT: total-liabilities-to-assets ratio: FD/AT = (DLTT+DLC)/AT
# FDBE: financial-debt-to-capital: measure of financial leverage - use book value as denominator
adata[, mean(at<=0 | is.na(at))]
adata[, mean(be<=0 | is.na(be))]

# Define leverage measures
# Notice that for 125 CHE can be negative
adata[at>0, fdat := tdebt/at]
adata[at>0 & che>0, fdat_net := (tdebt-che)/at]
adata[at>0, ltat := lt/at]
adata[be>0, fdbe := tdebt/be]
adata[be>0 & che>0, fdbe_net := (tdebt-che)/be]


# ================================================================= #
# Merge Compustat with CRSP link ####
# ================================================================= #
# -------------------------------------------------------------#
# Create COMP/PERMCO link table 
# datadate is not primary key of comp
comp_key <- adata[!is.na(datadate), .(gvkey,datadate)]
sum(duplicated(comp_key))

# Merge
setkey(comp_key, gvkey, datadate)
setkey(link, gvkey, linkdt, linkenddt)
linked <- merge(comp_key, link, by = 'gvkey')
linked <- linked[datadate >= linkdt][datadate <= linkenddt | is.na(linkenddt)]

# Permno/Datadate primary key
any(duplicated(linked[, .(datadate,permno)]))

# Tranforme primary to datadate/gvkey: drop for now possibility of two permco per gvkey for the same date
linked <- linked[!duplicated(linked[, .(datadate, gvkey, permco)])]
unique(linked[duplicated(linked[, .(gvkey, datadate)])]$gvkey)
linked <- linked[!duplicated(linked[, .(gvkey, datadate)])]
unique(linked[duplicated(linked[, .(gvkey, datadate)])]$gvkey)

# -------------------------------------------------------------#
# Add Permco
adata <- merge(adata, linked[, .(gvkey, datadate, permco)], by = c('gvkey', 'datadate'), all.x = TRUE)

# 61 % of annual compustat has a permco link
print(sprintf('Percentage of compustat rows with valid PERMCO: %.2f', 
              (adata[, mean(!is.na(permco))]*100)))
print(sprintf('Percentage of compustat rows with valid PERMCO (after 1990): %.2f', 
              (adata[fyear >= 1990, mean(!is.na(permco))]*100)))


# ================================================================= #
# Merge with CRSP names table ####
# ================================================================= #
stock_names <- stock_names[order(permco, nameenddt)]
stock_names <- unique(stock_names, by = c('permco', 'nameenddt'), fromLast = TRUE)

# Merge adata to names to industry codes
temp <- merge(adata[, .(permco, datadate)],
              stock_names,
              by = 'permco',
              allow.cartesian = TRUE)

temp <- temp[datadate>=namedt | is.na(namedt)][datadate<=nameenddt|is.na(nameenddt)]
temp <- temp[order(permco, nameenddt)]
temp <- unique(temp, by = c('permco', 'datadate', 'siccd'), fromLast = TRUE)
mean(duplicated(temp, by = c('permco', 'datadate'), fromLast = TRUE))
temp <- unique(temp, by = c('permco', 'datadate'), fromLast = TRUE)
temp[is.na(permco)]

adata <- merge(adata,
               temp[, .(permco, datadate, siccd, ticker)],
               by = c('permco', 'datadate'),
               all.x = TRUE)
rm(temp, stock_names)


# ================================================================= #
# Fill missing industry classification ####
# ================================================================= #
# Fill missing industry classification:
#   1. following observaion
#   2. names table
adata[, naicsh_filled := as.numeric(naicsh)]
adata[, sich_filled := as.numeric(sich)]

print(sprintf('Number of NAICSH missing %.2f%%. \n', adata[, mean(is.na(naicsh_filled))]*100)) # 38.60%. 
print(sprintf('Number of SICH missing %.2f%%. \n', adata[, mean(is.na(sich_filled))]*100)) # 48.42%.

# Complete sic from CRSP
adata[is.na(sich_filled), sich_filled := siccd]

print(sprintf('Number of NAICSH missing %.2f%%. \n', adata[, mean(is.na(naicsh_filled))]*100)) # 38.60%. 
print(sprintf('Number of SICH missing %.2f%%. \n', adata[, mean(is.na(sich_filled))]*100)) # 48.42%.

# Following ind classification
# na.locf(c(NA,1:10), na.rm=FALSE, fromLast=TRUE)
adata[,naicsh_filled := na.locf(naicsh_filled, na.rm = FALSE, fromLast = TRUE), by = gvkey]
adata[,sich_filled := na.locf(sich_filled, na.rm = FALSE, fromLast = TRUE), by = gvkey]

print(sprintf('Number of NAICSH still missing after backfilling: %.2f%%. \n',
              adata[, mean(is.na(naicsh_filled))]*100)) # 21.36%
print(sprintf('Number of SICH still missing after backfilling: %.2f%%. \n', 
              adata[, mean(is.na(sich_filled))]*100)) #  10.35%"

# adata[is.na(naicsh)]
# adata[gvkey=='328795', .(gvkey, datadate, sich, sich_filled, sic, naicsh, naicsh_filled, naics)]

# From names table fill either if has always been NA or if it is NA in the most recent data
# After checking, it does not happen that sich or naicsh becomes NA after it was not NA for some date.
# adata[is.na(naicsh_filled) & (datadate >= lnaics), unique(gvkey)]
adata[is.na(naicsh_filled), naicsh_filled := as.numeric(naics)]
adata[is.na(sich_filled), naicsh_filled := as.numeric(sic)]

print(sprintf('Final share of NAICSH still missing after filling from names table: %.2f%%. \n', 
              adata[, mean(is.na(naicsh_filled))]*100)) # 6.65%
print(sprintf('Final share of SICH still missing after filling from names table: %.2f%%. \n', 
              adata[, mean(is.na(sich_filled))]*100)) # 10.35%"

# Fill by GVKEY
adata[, naicsh_filled := na.locf(naicsh_filled, na.rm = FALSE, fromLast = TRUE), by = gvkey]
adata[, sich_filled := na.locf(sich_filled, na.rm = FALSE, fromLast = TRUE), by = gvkey]
adata[, naicsh_filled := na.locf(naicsh_filled, na.rm = FALSE, fromLast = FALSE), by = gvkey]
adata[, sich_filled := na.locf(sich_filled, na.rm = FALSE, fromLast = FALSE), by = gvkey]

# Fill by PERMCO
adata[!is.na(permco), naicsh_filled := na.locf(naicsh_filled, na.rm = FALSE, fromLast = TRUE), by = permco]
adata[!is.na(permco), sich_filled := na.locf(sich_filled, na.rm = FALSE, fromLast = TRUE), by = permco]
adata[!is.na(permco), naicsh_filled := na.locf(naicsh_filled, na.rm = FALSE, fromLast = FALSE), by = permco]
adata[!is.na(permco), sich_filled := na.locf(sich_filled, na.rm = FALSE, fromLast = FALSE), by = permco]

print(sprintf('Final share of NAICSH still missing: %.2f%%. \n', 
              adata[, mean(is.na(naicsh_filled))]*100)) # 8.21% or 9.70%
print(sprintf('Final share of SICH still missing after: %.2f%%. \n', 
              adata[, mean(is.na(sich_filled))]*100)) # 0%"

# -------------------------------------------------------------#
# Industry codes
ind_vars <- c('sich_filled', 'naicsh_filled')
adata[, (paste0(ind_vars,'_2')) := lapply(.SD, function(x){as.numeric(substring(x,1,2))}), .SDcols = ind_vars]
adata[, (paste0(ind_vars,'_3')) := lapply(.SD, function(x){as.numeric(substring(x,1,3))}), .SDcols = ind_vars]
adata[, (paste0(ind_vars,'_4')) := lapply(.SD, function(x){as.numeric(substring(x,1,4))}), .SDcols = ind_vars]


# ================================================================= #
# Write data ####
#================================================================== #
adata[, num := .N, by = .(gvkey, datadate)]
check <- adata[num >= 2]
adata <- unique(adata, by = c('gvkey', 'datadate'))

if(any(duplicated(adata[, .(gvkey, datadate)]))){
  stop('Data primary key violated.')
}

# save down data
#saveRDS(adata, file = paste0(PROCDIR, 'Compustat/compustat_annual.rds'))
fwrite(adata, file = paste0(PROCDIR, "Compustat/compustat_annual.csv"))
print('Compustat annual has been created successfully.')



