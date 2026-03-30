# ================================================================= #
# compustat_quarterly.R ####
# ================================================================= #
# Description:
# ------------
#   Create compustat quarterly (processed data) data set.
#   It includes a bunch of variables like be, leverage ratios, ind concentration, etc.
#   It also includes the merge with some annual variables.
#
# Input(s):
# ---------
#   1. Compustat quarterly raw data:
#     data/raw/compustat/compustat_fundq.rds
#
#   2. Compustat annual processed data:
#     data/processed/compustat/compustat_annual.RDS
#
# Output(s):
# ----------
#   Compustat quarterly processed data:
#     data/processed/compustat/compustat_quarterly.rds
#
# Date:
# ----------
#   2019-08-26
#   update: 2026-01-23
#
# Author(s):
# ----------
#   Lira Mota, liramota@mit.edu
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
print('Started script to compustat quarterly.')

# Import libraries
library(data.table) 
library(zoo)
library(lubridate)
library(ggplot2)

# Source helper scripts
source("utils/setPaths.R")
source("utils/f_compustat_variables_calculations.R")

start_time <- Sys.time()
print('Start Compustat quartely computation.')


# ================================================================= #
# Read data ####
# ================================================================= #
# (Raw) Compustat Quarterly
comp <- readRDS(paste0(RAWDIR, 'Compustat/compustat_fundq.rds'))

# (Processed) Compustat Annual
# Variables downloaded from annual table - not available quarterly
# Always identified with (yy) after merge
compa <- fread(paste0(PROCDIR, 'Compustat/compustat_annual.csv'))
compa <- compa[, .(gvkey, datadate, fyear, permco, 
                   sicf = sich_filled, naicsf = naicsh_filled, state, city, county,
                   aqc, at, capx, ceq, che, gp, ebit, ebitda, ib, itcb, ivao, 
                   lt, mib, pstk, pstkl, pstkrv, revt, seq, txdb, txditc, xlr, be)]
# -------------------------------------------------------------#
print(sprintf('Data import complete in %.2f.', Sys.time() - start_time))
start_time_inter <- Sys.time()


# ================================================================= #
# Data cleaning ####
# ================================================================= #
# Check for for duplicates
comp[, mdate := year(datadate)*100 + month(datadate)] # we need mdate and gvkey to be primary key

## For now drop duplicates - it can happen because company changed the end fiscal year data
print(sprintf('Rows dropped because of missing fyear: %.0f', 
              (comp[,sum(is.na(datafqtr))])))
comp <- comp[!is.na(datafqtr)] # Droped 1075 comp[,sum(is.na(datafqtr))]  
comp <- comp[!is.na(datacqtr)] # Droped 2695 comp[,sum(is.na(datacqtr))]  

print(sprintf('Rows dropped because of duplicated (gvkey, datadate): %.0f', 
              (sum(duplicated(comp,by=c('gvkey', 'datadate'))))))

print(sprintf('Rows dropped because of duplicated (gvkey, datacqtr): %.0f', 
              (sum(duplicated(comp,by=c('gvkey', 'datacqtr'))))))

print(sprintf('Rows dropped because of duplicated (gvkey, datafqtr): %.0f', 
              (sum(duplicated(comp,by=c('gvkey', 'datafqtr'))))))

comp <- comp[order(gvkey, datafqtr, datadate)]
comp <- comp[!duplicated(comp[, .(gvkey,datadate)], fromLast = T)]
comp <- comp[!duplicated(comp[, .(gvkey,mdate)], fromLast = T)]
comp <- comp[!duplicated(comp[, .(gvkey,datafqtr)], fromLast = T)] # 25 cases


# ================================================================= #
# Merge with annual processed data ####
# ================================================================= #
comp[is.na(gvkey) | is.na(fyearq)]
compa[is.na(gvkey) | is.na(fyear)]

colnames(compa)[-c(1:9)] <- paste0(colnames(compa)[-c(1:9)],'yy')
compa[,datadate := NULL]
compa[, gvkey := as.character(gvkey)]
comp <- merge(comp, compa, 
              by.x = c('gvkey', 'fyearq'), 
              by.y = c('gvkey', 'fyear'),
              all.x = TRUE)


# ================================================================= #
# Create complete panel ####
# ================================================================= #
# Create Fiscal Date 
any(duplicated(comp, by = c('gvkey', 'datafqtr')))
comp[,qdate := as.yearqtr(datafqtr, '%YQ%q')]

# Could not complete before due to fyearq na's that would generate bad merges.
# Notice that qdate is datafqtr (fiscal quarter)
bpanel <- CJ(unique(comp$gvkey), unique(comp$qdate))
colnames(bpanel) <- c('gvkey', 'qdate')

comp <- comp[order(gvkey, qdate)]
bpanel <- merge(bpanel, comp, by = c('qdate', 'gvkey'), all.x = TRUE)

bpanel <- merge(bpanel,
                comp[, .(first_quater = min(datafqtr), 
                         last_quarter = max(datafqtr)), 
                     by = gvkey],
                by = 'gvkey')
bpanel <- bpanel[(qdate>=first_quater & qdate<=last_quarter)]
comp <- copy(bpanel)
comp <- comp[order(gvkey, qdate)]
comp[, c('first_quater', 'last_quarter') := NULL]
rm(bpanel)
any(duplicated(comp, by = c('gvkey', 'qdate')))


# ================================================================= #
# Transform year to date variables into quarterly ####
# ================================================================= #
any(duplicated(comp, by = c('gvkey', 'qdate')))
#colnames(comp)

# Fill NA's created from full panel.
comp[is.na(qdate)]
comp[, fyearq := as.integer(fyearq)]
comp[, fqtr := as.integer(fqtr)]
comp[is.na(fyearq), fyearq := year(qdate)]
comp[is.na(fqtr), fqtr := as.integer(quarter(qdate))]
comp <- comp[order(gvkey, fyearq, fqtr)]

#comp[!is.na(rdipy)&rdipy!=0,.(qdate, gvkey, rdipy, rdipq)]
#comp[gvkey=='022330' & year(qdate)>=2002,.(qdate, gvkey, rdipy, rdipq) ]
#comp[is.na(rdipy)&!is.na(rdipq)&rdipq!=0,.(qdate, gvkey, rdipy, rdipq)]
#comp[is.na(rdipq)&!is.na(rdipy)&rdipy!=0,.(qdate, gvkey, rdipy, rdipq)]

# Annualized variables
pvars <- c('aqcy', 'capxy', 'chechy', 'depcy', 'dlcchy', 'dltisy','dltry', 'dvy', 'exrey',
           'fiaoy', 'fincfy', 'ivacoy', 'ivncfy', 'ivchy', 'ivstchy', 'nopiy', 'oancfy',
           'prstkcy', 'prstkccy', 'sivy', 'sppey','sstky', 'txbcofy', 'wcapchy')

# New Vars
nvars <- c('aqcq', 'capxq', 'chechq', 'depcq', 'dlcchq', 'dltisq','dltrq', 'dvq', 'exreq',
           'fiaoq', 'fincfq', 'ivacoq', 'ivncfq', 'ivchq', 'ivstchq', 'nopiq', 'oancfq', 
           'prstkcq', 'prstkccq', 'sivq', 'sppeq','sstkq', 'txbcofq', 'wcapchq')

# Fill NA's
fillNAgaps <- function(x, firstBack=FALSE) {
  ## NA's in a vector or factor are replaced with last non-NA values
  ## If firstBack is TRUE, it will fill in leading NA's with the first
  ## non-NA value. If FALSE, it will not change leading NA's.
  
  # If it's a factor, store the level labels and convert to integer
  lvls <- NULL
  if (is.factor(x)){
    lvls <- levels(x)
    x <- as.integer(x)
  }
  
  goodIdx <- !is.na(x)
  
  # These are the non-NA values from x only
  # Add a leading NA or take the first good value, depending on firstBack   
  if (firstBack){
    goodVals <- c(x[goodIdx][1], x[goodIdx])
  }
  else{
    goodVals <- c(NA, x[goodIdx])
  }            
  
  # Fill the indices of the output vector with the indices pulled from
  # these offsets of goodVals. Add 1 to avoid indexing to zero.
  fillIdx <- cumsum(goodIdx)+1
  
  x <- goodVals[fillIdx]
  
  # If it was originally a factor, convert it back
  if (!is.null(lvls)) {
    x <- factor(x, levels=seq_along(lvls), labels=lvls)
  }
  return(x)
}

# Calculate difference taking care of NA's
fill_values <- function(x){
  y <- fillNAgaps(x)
  y <- y - shift(y)
  m1 <- is.na(y) & !is.na(x)
  y[m1] <- x[m1]
  m2 <- is.na(x) & y == 0
  y[m2] <- as.numeric(NA)
  y
}

comp[,(nvars):=lapply(.SD, fill_values), .SDcols= pvars, by=.(gvkey, fyearq)]
# Check if the total is right
#comp[!is.na(capxy),.(capxy=last(capxy),capxqt=sum(capxq, na.rm=TRUE)), by = .(gvkey, fyearq)][round(capxy,3)!=round(capxqt,3)]
# Apple look
# ggplot(comp[gvkey=='001690'], aes(x=as.Date(as.yearqtr(datacqtr, format = "%YQ%q"))))+
#   #geom_line(aes(y=capxy, colour='capxy'))+
#   #geom_line(aes(y=capxq, colour='capxq'))
# # Total - notice that capx only stops being missing after the 80's
# ggplot(comp[ ,.(capxy=mean(capxy, na.rm=T),
#                 capxq=mean(capxq, na.rm=T)),
#              by=datacqtr],
#        aes(x=as.Date(as.yearqtr(datacqtr, format = "%YQ%q"))))+
#    geom_line(aes(y=capxy, colour='capxy'))+geom_line(aes(y=capxq, colour='capxq'))
# View(comp[gvkey=='001690',.(gvkey,fyearq, fqtr, capxy,capxq)])
# View(comp[gvkey=='328795',.(gvkey,fyearq, fqtr, capxy,capxq)])

# -------------------------------------------------------------#
print(sprintf('Year to date to quartely complete in %.2f.', 
              Sys.time() - start_time_inter))
start_time_inter <- Sys.time()

# ================================================================= #
# Create quarterly variables of interest ####
# ================================================================= #
# Assets
comp[, atq_lag := shift(atq), by = .(gvkey)]
comp[atq_lag > 1 , asset_growth := (atq / atq_lag) - 1]

# -------------------------------------------------------------#
# Book Value (Quarterly)
comp[, beq := calculate_be(comp, freq = 'quarterly')]
comp[, beq_lag := shift(beq), by = .(gvkey)]

# -------------------------------------------------------------#
# Total Debt
comp[, 'tdebtq' := sum(dlttq, dlcq, na.rm = TRUE), by = 1:NROW(comp)]

# -------------------------------------------------------------#
# Leverage 
# http://onlinelibrary.wiley.com/doi/10.1111/j.1468-2443.2010.01125.x/epdf
# FDAT: financial-debt-to-assets ratio: FD/AT = (DLTT+DLC)/AT: Welch argues it is a wrong measure to analyze capital structure
# LTAT: total-liabilities-to-assets ratio: FD/AT = (DLTT+DLC)/AT
# FDBE: financial-debt-to-capital: measure of financial leverage - use book value as denominator
comp[, mean(atq<=0 | is.na(atq))]
comp[, mean(beq<=0 | is.na(beq))]

# Define leverage measures
comp[, c('fdat', 'fdat_net', 'ltat', 'fdbe', 'fdbe_net') := numeric()]
comp[atq > 0, fdatq := (dlttq+dlcq) / atq]
comp[atq > 0, fdatq_net := (dlttq+dlcq-cheq) / atq]
comp[atq > 0, ltatq := ltq / atq]
comp[atq > 0, ltfdatq := dlttq / atq]
comp[beq > 0, fdbeq := (dlttq+dlcq) / beq]
comp[beq > 0, fdbeq_net := (dlttq+dlcq-cheq) / beq]
#View(comp[gvkey=='005073',.(cyqtr,BE,at,che,dltt,dlc,fdat,fdat_net,ltat,fdbe,fdbe_net)])

# -------------------------------------------------------------#
# Operating Profits 
comp[, 'costtq' := sum(cogsq, xsgaq, xintq, na.rm = TRUE), by = 1:NROW(comp)]
comp[is.na(cogsq) & is.na(xsgaq) & is.na(xintq), costtq := NA]
summary(comp$costtq) #TODO: what is negative cost?
comp[, 'opq' := saleq - costtq]
comp[atq > 0, 'opq_atq' := opq / atq]

# -------------------------------------------------------------#
# EBTA
comp[, 'costq' := sum(cogsq, xsgaq, na.rm = TRUE), by = 1:NROW(comp)]
comp[is.na(cogsq) & is.na(xsgaq), cost := NA]
summary(comp$costq) #TODO: what is negative cost?
comp[, 'ebitaq' := saleq - costq]
summary(comp$ebitaq)

# -------------------------------------------------------------#
# Tobin's Q
comp[, meq_comp := (prccq * cshoq)]
comp[beq > 0, tobq:= meq_comp / beq]
summary(comp$tobq)
comp[, mean(is.na(tobq))]

# -------------------------------------------------------------#
# CHE
comp[atq > 0, cheq_atq := cheq / atq]

# -------------------------------------------------------------#
# Operating profits
comp[atq > 0, oibdpq_atq := oibdpq / atq]

# -------------------------------------------------------------#
# Interest rate coverage ratio
comp[ebitaq > 0 & xintq >= 0, icr := xintq / ebitaq]
comp[saleq > 0 & xintq >= 0,  icrsale := xintq / saleq]

# -------------------------------------------------------------#
print(sprintf('Quartely variable calculations complete in %.2f.', 
              (Sys.time() - start_time_inter)/60))
start_time_inter <- Sys.time()


# ================================================================= #
# Compute cash flow variables ####
# ================================================================= #
# -------------------------------------------------------------#
# Cash flow identity 
# Total net debt issuance
comp[, stndiq := calculate_stndi(comp, freq = "quarterly")]
comp[, ndiq := calculate_ndi(comp, freq = "quarterly")]
comp[, tndiq := stndiq + ndiq]

# Total payouts
comp[, nerq := calculate_ner(comp, freq = "quarterly")]
comp[, cdvq := calculate_cdv(comp, freq = "quarterly")]
comp[, payoutq := cdvq + nerq]

# Total investment
comp[, kinvq := calculate_kinv(comp, freq = "quarterly")]
comp[, taqcq := calculate_taqc(comp, freq = "quarterly")] ##SDC
comp[, iinvq := calculate_iinv(comp, freq = "quarterly")]
comp[, rinvq := kinvq + taqcq + iinvq]
comp[, finvq := calculate_finv(comp, freq = "quarterly")]
comp[, tinvq := rinvq + finvq]

# Net operating profits
comp[, nopq := calculate_nop(comp, freq = "quarterly")]

# Other expenses
comp[, oexq := calculate_oex(comp, freq = "quarterly")]

# Checks 
comp[, money_inq := nopq + tndiq]
comp[, money_outq := payoutq + tinvq + chechq + oexq]
comp[, diff_in_outq := round(money_inq - money_outq, 2)]

sprintf('--- Summary of error in cash flow identity -----')
comp[, summary(diff_in_outq)]
sprintf('Error is different than 0 for %.2f percent of the data.', 
        comp[, mean(diff_in_outq!=0, na.rm = TRUE)]*100)
sprintf('Error in absolute value is larger than $100 Mi for %.2f percent of the data.', 
        comp[, mean(abs(diff_in_outq)>100, na.rm = TRUE)]*100)

sprintf('--- Summary of other expenses -----')
comp[money_inq !=0 , summary(oexq / money_inq)]


# ================================================================= #
# Create calendar date ####
# ================================================================= #
# Think what is the date variable I am interested in. 
# Fiscal quarter can be very different than calendar quarter. 
# Calendar must be my reference date.
#comp[is.na(datafqtr)]
#comp[is.na(datacqtr)] # 953 missings

# Create calendar quarter variable: Imputing from datadate does not work, 
# creates duplicates.
comp[, cyqtr := as.yearqtr(datacqtr,'%YQ%q')]
comp <- comp[!is.na(cyqtr)]
comp[, qdate := NULL]
# Check duplicates
any(duplicated(comp, by = c('gvkey', 'cyqtr')))

# -------------------------------------------------------------#
sprintf('Cumulate flow/growth varaibles complete in %s.', 
        Sys.time() - start_time_inter) 


# ================================================================= #
# Fill missing industry classification ####
# ================================================================= #
# Fill missing industry classification:
#   1. following observaion
#   2. names table
print(sprintf('Number of NAICSF missing %.2f%%. \n', comp[, mean(is.na(naicsf))]*100)) # 38.60%. 
print(sprintf('Number of SICF missing %.2f%%. \n', comp[, mean(is.na(sicf))]*100)) # 48.42%.

# Following ind classification
# na.locf(c(NA,1:10), na.rm=FALSE, fromLast=TRUE)
comp[, naicsf := na.locf(naicsf, na.rm = FALSE, fromLast = TRUE), by = gvkey]
comp[, naicsf := na.locf(naicsf, na.rm = FALSE, fromLast = FALSE), by = gvkey]

comp[, sicf := na.locf(sicf, na.rm = FALSE, fromLast = TRUE), by = gvkey]
comp[, sicf := na.locf(sicf, na.rm = FALSE, fromLast = FALSE), by = gvkey]

print(sprintf('Number of NAICSF still missing after backfilling: %.2f%%. \n',
              comp[, mean(is.na(naicsf))]*100))
print(sprintf('Number of SICF still missing after backfilling: %.2f%%. \n', 
              comp[, mean(is.na(sicf))]*100)) 

# Fill by PERMCO
comp[!is.na(permco), naicsf := na.locf(naicsf, na.rm = FALSE, fromLast = TRUE), by = permco]
comp[!is.na(permco), naicsf := na.locf(naicsf, na.rm = FALSE, fromLast = FALSE), by = permco]

comp[!is.na(permco), sicf := na.locf(sicf, na.rm = FALSE, fromLast = TRUE), by = permco]
comp[!is.na(permco), sicf := na.locf(sicf, na.rm = FALSE, fromLast = FALSE), by = permco]

print(sprintf('Final share of NAICSF still missing after backfilling: %.2f%%. \n',
              comp[, mean(is.na(naicsf))]*100)) # 2.90%
print(sprintf('Final share of SICF still missing after backfilling: %.2f%%. \n', 
              comp[, mean(is.na(sicf))]*100)) #  6.54%"%"


# ================================================================= #
# Write Data ####
# ================================================================= #
if(any(duplicated(comp[, .(cyqtr, gvkey)]))){
  stop('Compustat data primary key violated.')
}

if(any(duplicated(comp[, .(mdate, gvkey)]))){
  stop('Compustat data primary key violated.')
}

saveRDS(comp, paste0(PROCDIR, 'Compustat/compustat_quarterly.rds'))
fwrite(comp, paste0(PROCDIR, 'Compustat/compustat_quarterly.csv'))
print('Compustat quarterly has been created successfully.')


