# ================================================================= #
# import_fundq.R ####
# ================================================================= #
# Description:
# ------------
#   This downloads and saves Compustat quarterly data from WRDS server.
#
# Input(s):
# ---------
#   WRDS connection
#
# Output(s):
# ----------
#   Compustat quarterly raw data:
#     compustat_fundq
#
# Date:
# ----------
#   2019-08-28
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
#     Compustat XpressFeed Variables:
#     * AQC(Y)  = Acquisitions
#     * AT      = Total Assets  
#     * CAPX    = Capital Expenditures  
#     * CEQ     = Common/Ordinary Equity - Total 
#     * COGS    = Cost of Goods Sold
#     * CSHO    = Common Shares Outstanding
#     * DEPC    = Depreciation and Depletion (Cash Flow)
#     * DLC     = Debt in Current Liabilities 
#     * DLCCH   = Current Debt - Changes
#     * DLTIS   = Long-Term Debt - Issuance
#     * DLTT    = Long-Term Debt - Total
#     * DLTR    = Long-Term Debt - Reduction
#     * DP      = Depreciation and Amortization (Income statement)
#     * DV      = Cash Dividends (Cash Flow)  
#     * DVC     = Divdends common/ordinary  
#     * DVP     = Dividends - Preferred/Preference
#     * EBIT    = Earnings Before Interest and Taxes
#                 This item is the sum of Sales - Net (SALE) minus Cost of Goods Sold (COGS) 
#                 minus Selling, General & Administrative Expense (XSGA) minus Depreciation/Amortization (DP). 
#     * EBITDA  = Earnings Before Interest
#                 This item is the sum of Sales - Net (SALE) minus Cost of Goods Sold (COGS) 
#                 minus Selling, General & Administrative Expense (XSGA).
#     * GDWLQ   = Goodwill
#     * GP      = Gross Profits
#     * IB      = Income Before Extraordinary Items
#     * ICAPT   = Invested Capital - Total
#     * INTAN   = Intangible Assets - Total
#     * ITCB    = Investment Tax Credit (Balance Sheet)  
#     * LCT     = Current Liabilities - Total        
#     * LT      = Total Liabilities        
#     * MIB     = Minority Interest (Balance Sheet) 
#     * NAICS   = North American Industrial Classification System Variable Name
#     * NAICSH  = North American Industry Classification Codes - Historical Company Variable Name
#     * NI      = Net Income
#     * OIBDPQ  = Operating Income Before Depreciation
#     * PPEGT   = Property, Plant and Equipment - Total (Gross)
#     * PPENT   = Property, Plant, and Equipment Construction in Progress (Net)
#     * PRBA    = Postretirement Benefit Assets (from separate pension annual file) 
#     * PRCCQ	  =	Price Close - Quarter
#     * PRSTKC  = Purchase of Common and Preferred Stock                   
#     * PRSTKCC = Purchase of Common Stock (Cash Flow)                   
#     * PSTKRV  = Preferred Stock Redemption Value   
#     * PSTK    = Preferred/Preference Stock (Capital) - Total (kd: = par?)               
#     * PSTKL   = Preferred Stock Liquidating Value     
#     * PSTKRV  = Preferred Stock Liquidating Value       
#     * OIADPQ  = Operating Income After Depreciation - Quarterly
#     * OIBDPQ  = Operating Income After Depreciation - Quarterly
#     * RDIP    = In Process R&D
#     * RE      = Retained Earnings
#     * REVT    = Revenue - Total
#     * SALE    = Sales/Turnover Net
#     * SEQ     = Shareholders Equity   
#     * SIC     = Standard Industrial Classification Code 
#     * SSTK    = Sale of Common and Preferred Stock 
#     * TXDB    = Deferred Taxes Balance Sheet
#     * TXDI    = Income Taxes - Deferred
#     * TXDITC  = Deferred Taxes and Investment Tax Credit                        
#     * WCAPCH  = Working Capital Change - Total
#     * XINT    = Interest and Related Expense - Total 
#     * XLR     = Staff Expense - Total
#     * XRD     = Research and Development Expense 
#     * XSGAQ   = Selling, General and Administrative Expenses (millions) 
#
# ================================================================= #


# ================================================================= #
# Environment ####
# ================================================================= #
# Clear workspace
rm(list = ls())

# Import libraries
library(data.table)
library(RPostgres)

# Source helper scripts
source('utils/setPaths.R')
source('utils/wrds_credentials.R')

# Create database connections
creds <- get_wrds_credentials()
wrds <- dbConnect(Postgres(),
                  host='wrds-pgdata.wharton.upenn.edu',
                  port=9737,
                  user = creds$username,
                  password = creds$password,
                  sslmode='require',
                  dbname='wrds')


# ================================================================= #
# Read data ####
# ================================================================= #
print('Sucessfully started downloading Compustat quarterly.')

# List of variables to be downloaded
varlist <- c("cusip", "tic", "cik", "conm",
             "datacqtr", "fyearq", "fqtr", "datafqtr", "fyr",
             "aqcy", "atq", "capxy", "ceqq", "chq", "cheq", "chechy", 
             "cogsq", "cshoq","cshopq", "depcy", 
             "dlcq",  "dlcchy", "dltisy", 
             "dlttq", "dltry", "dpq", "dvy", "exrey", "fiaoy", "fincfy", "gdwlq", 
             "icaptq", "intanq", "ivacoy", "ivchy", "ivncfy", "ivstchy", "lctq", "ltq", "mibq", 
             "niq", "nopiy", "oancfy", "oibdpq", "oiadpq", "prccq", "pdateq", "ppentq", "ppegtq", 
             "prstkcy", "prstkccy", "pstkq", "req", "rdipq", "rdipy", "saleq", 
             "seqq", "sivy", "sppey", "sstky", "txbcofy", "txdbq", "txdiq", "txditcq", 
             "wcapchy", "xintq", "xrdq", "xsgaq")

# SQL query to WRDS connection
compq <- dbGetQuery(wrds, 
                    sprintf("SELECT %s FROM COMP.FUNDQ
                             WHERE indfmt = 'INDL'
                             AND datafmt = 'STD'
                             AND popsrc = 'D'
                             AND consol = 'C'", 
                            paste(c('gvkey', 'datadate', varlist), collapse = ',')))


# ================================================================= #
# Clean and organize data ####
# ================================================================= #
# Create data.table and sort data
compq <- as.data.table(compq)
compq <- compq[order(gvkey, datadate, fyearq)]

res <- dbSendQuery(wrds, "select column_name
                   from information_schema.columns
                   where table_schema='comp'
                   and table_name='fundq'
                   order by column_name")
data <- dbFetch(res, n=-1)
dbClearResult(res)
data


# ================================================================= #
# Check data integrity ####
# ================================================================= #
# Is c(GVKEY, datadate) primary key?
check <- nrow(compq[is.na(gvkey)|is.na(datadate)])==0
if(!check){print('GVKEY or DATADATE missing data.')}

check <- nrow(unique(compq, v=v('gvkey', 'datadate'))) == nrow(compq)
if(!check){print('GVKEY or DATADATE is NOT primary key.')}

if (any(duplicated(compq, by = c('gvkey', 'datadate')))){
  print("Duplicates in primary keys.")
}


# ================================================================= #
# Write data ####
# ================================================================= #
# save data
saveRDS(compq, paste0(RAWDIR, 'compustat/compustat_fundq.rds'))
print('Compustat quarterly data has been downloaded to database successfully.')




