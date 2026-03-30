# ================================================================= #
# import_funda.R ####
# ================================================================= #
# Description:
# ------------
#   This downloads and saves Compustat annual data from WRDS server.
#   It uses RPostgres direct download.
#
# Input(s):
# ---------
#   WRDS connection
#
# Output(s):
# ----------
#   Compustat annual raw data:
#     compustat_funda
#
# Date:
# ----------
#   2022-05-12
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
# Compustat XpressFeed Variables:
# * AQC(Y)  = Acquisitions
# * AT      = Total Assets  
# * CAPX    = Capital Expenditures  
# * CEQ     = Common/Ordinary Equity - Total 
# * COGS    = Cost of Goods Sold
# * CSHO    = Common Shares Outstanding
# * DLC     = Debt in Current Liabilities 
# * DLCCH   = Current Debt - Changes
# * DLTIS   = Long-Term Debt - Issuance
# * DLTT    = Long-Term Debt - Total
# * DLTR    = Long-Term Debt - Reduction
# * DP      = Depreciation and Amortization
# * DVC     = Divdends common/ordinary  
# * DVP     = Dividends - Preferred/Preference
# * EBIT    = Earnings Before Interest and Taxes
#             This item is the sum of Sales - Net (SALE) minus Cost of Goods Sold (COGS) minus Selling, General & Administrative Expense (XSGA) minus Depreciation/Amortization (DP). 
# * EBITDA  = Earnings Before Interest
#             This item is the sum of Sales - Net (SALE) minus Cost of Goods Sold (COGS) minus Selling, General & Administrative Expense (XSGA).
# * GP      = Gross Profits
# * IB      = Income Before Extraordinary Items
# * ICAPT   = Invested Capital - Total
# * ITCB    = Investment Tax Credit (Balance Sheet)  
# * LT      = Total Liabilities        
# * MIB     = Minority Interest (Balance Sheet) 
# * NAICS   = North American Industrial Classification System Variable Name
# * NAICSH  = North American Industry Classification Codes - Historical Company Variable Name
# * NI      = Net Income
# * OIBDPQ  = Operating Income Before Depreciation
# * PPEGT   = "Property, Plant and Equipment - Total (Gross)"
# * PRBA    = Postretirement Benefit Assets (from separate pension annual file) 
# * PRCC_F	= Float	Price Close - Annual - Fiscal (prcc_f)
# * PRSTKC  = Purchase of Common and Preferred Stock                   
# * PRSTKCC = Purchase of Common Stock (Cash Flow)                   
# * PSTKRV  = Preferred Stock Redemption Value   
# * PSTK    = Preferred/Preference Stock (Capital) - Total (kd: = par?)               
# * PSTKL   = Preferred Stock Liquidating Value     
# * PSTKRV  = Preferred Stock Liquidating Value          
# * RE      = Retained Earnings
# * REVT    = Revenue - Total
# * SALE    = Sales/Turnover Net
# * SEQ     = Shareholders Equity   
# * SIC     = Standard Industrial Classification Code 
# * SSTK    = Sale of Common and Preferred Stock 
# * TXDB    = Deferred Taxes Balance Sheet
# * TXDI    = Income Taxes - Deferred
# * TXDITC  = Deferred Taxes and Investment Tax Credit                        
# * WCAPCH  = Working Capital Change - Total
# * XINT    = Interest and Related Expense - Total 
# * XLR     = Staff Expense - Total
# * XRD     = Research and Development Expense 
# * XSGA   = Selling, General and Administrative Expenses (millions) 
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
print('Successfully started downloading compustat annually from WRDS.')

# List of variables to be downloaded
varlist <- c('cusip', 'conm', 'tic', 'fdate',
             'aqc', 'at','capx', 'ceq','ch','che', 'cik',
             'cogs', 'csho',
             'dlc', 'dlcch',
             'dltis', 'dltt', 'dltr',
             'dp', 'dv', 'dvc', 'dvt', 'dvp',
             'ebit', 'ebitda', 'emp', 'gp',
             'ib', 'icapt', 'itcb',
             'ivao', 'lt', 'mib',
             'naicsh',
             'ni', 'oibdp', 'pdate',
             'ppegt', 'prstkc', 'prstkcc', 'pstk', 'pstkl', 'pstkrv',
             're', 'revt', 'sale', 'seq', 'sich',
             'scstkc', 'sstk', 
             'txdb', 'txdi', 'txditc', 'wcapch', 'xint', 'xlr',
             'xrd', 'xsga', 'prcc_f')

# SQL query to WRDS connection
sql <- sprintf( "SELECT %s FROM COMP.FUNDA
                WHERE indfmt = 'INDL'
                AND datafmt = 'STD'
                AND popsrc = 'D'
                AND consol = 'C'", 
                paste(c('gvkey', 'datadate', 'fyear', varlist), collapse = ','))
adata <- dbGetQuery(wrds, sql)


# ================================================================= #
# Clean and organize data ####
# ================================================================= #
# Create data.table and sort data
adata <- as.data.table(adata)
adata <- adata[order(gvkey, datadate, fyear)]


#================================================================== #
# Check data integrity ####
#================================================================== #
# Is c(GVKEY, datadate) primary key?
check <- nrow(adata[is.na(gvkey)|is.na(datadate)])==0
if(!check){print('GVKEY or DATADATE missing data.')}

check <- nrow(unique(adata, v=v('gvkey', 'datadate'))) == nrow(adata)
if(!check){print('GVKEY or DATADATE is NOT primary key.')}

if (any(duplicated(adata, by = c('gvkey', 'datadate')))){
  print("Duplicates in primary keys.")
}

# Set primary keys
setkeyv(adata, c('gvkey', 'datadate'))


#================================================================== #
# Write data ####
#================================================================== #
if(any(duplicated(adata[, .(gvkey, datadate)]))){
  stop('Data primary key violated.')
}

# save data
saveRDS(adata, paste0(RAWDIR, 'Compustat/compustat_funda.rds'))
print('Compustat annual data has been downloaded to database successfully.')




