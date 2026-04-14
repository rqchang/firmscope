# ================================================================= #
# import_comp_axiliaries.R ####
# ================================================================= #
# Description:
# ------------
#   This downloads and saves Compustat auxiliary tables.
#   It uses RPostgres direct download.
#
# Input(s):
# ---------
#   WRDS connection
#
# Output(s):
# ----------
#   Compustat names raw data:
#     compustat_names
#     compustat_histnamess
#     compustat_conml
#
# Date:
# ----------
#   2022-05-12
#   update: 2026-03-30
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
print('Successfully started downloading compustat auxiliaries from WRDS.')

# Import and Save Names Table ####
vars <- dbGetQuery(wrds, "
  SELECT column_name
  FROM information_schema.columns
  WHERE table_schema = 'crsp'
    AND table_name = 'comphist'
  ORDER BY column_name
")

comp_histnames <- dbGetQuery(wrds,"SELECT * from crsp.comphist")

# Download Compustat names table: get sic/naics code (most recent one) ####
comp_names <- dbGetQuery(wrds,"SELECT * from comp.names")

# Download Compustat company legal names and locations ####
vars <- dbGetQuery(wrds, "
  SELECT column_name
  FROM information_schema.columns
  WHERE table_schema = 'comp'
    AND table_name = 'company'
  ORDER BY column_name
")

company_vars <- c(
  "gvkey", "conm", "conml",
  "state", "city", "county",
  "loc", "fic",
  "sic", "naics",
  "gsector", "ggroup", "gind", "gsubind",
  "cik", "ipodate"
)

company <- dbGetQuery(wrds, sprintf("SELECT %s FROM comp.company", paste(company_vars, collapse = ",")))


# ================================================================= #
# Clean and organize data ####
# ================================================================= #
# Create data.table and sort data
comp_histnames <- as.data.table(comp_histnames)
comp_names <- as.data.table(comp_names)
company <- as.data.table(company)


#================================================================== #
# Check data integrity ####
#================================================================== #
# ---------------------------------------------- #
# comp_names
# ---------------------------------------------- #
# Is c(GVKEY) primary key?
check <- nrow(comp_names[is.na(gvkey)])==0
if(!check){print('GVKEY missing data.')}

check <- nrow(unique(comp_names, v=v('gvkey'))) == nrow(comp_names)
if(!check){print('GVKEY is NOT primary key.')}

if (any(duplicated(comp_names, by = c('gvkey')))){
  print("Duplicates in primary keys.")
}

# Set primary keys
setkeyv(comp_names, c('gvkey'))

# ---------------------------------------------- #
# company
# ---------------------------------------------- #
# Is c(GVKEY) primary key?
check <- nrow(company[is.na(gvkey)])==0
if(!check){print('GVKEY missing data.')}

check <- nrow(unique(company, v=v('gvkey'))) == nrow(company)
if(!check){print('GVKEY is NOT primary key.')}

if (any(duplicated(company, by = c('gvkey')))){
  print("Duplicates in primary keys.")
}

# Set primary keys
setkeyv(company, c('gvkey'))


#================================================================== #
# Write data ####
#================================================================== #
# save files
saveRDS(comp_names, paste0(RAWDIR, 'Compustat/compustat_names.rds'))
saveRDS(comp_histnames, paste0(RAWDIR, 'Compustat/compustat_histnames.rds'))
saveRDS(company, paste0(RAWDIR, 'Compustat/compustat_conml.rds'))
print('Compustat names data has been downloaded to database successfully.')



