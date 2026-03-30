#=======================================================================================#
# WRDS Credentials ####
#=======================================================================================#
# In R, run once:
#file.edit("~/.Renviron")

# Add exactly (one per line, no quotes):
#WRDS_USERNAME=your_wrds_username
#WRDS_PASSWORD=your_wrds_password

get_wrds_credentials <- function() {
  wrds_username <- Sys.getenv("WRDS_USERNAME")
  wrds_password <- Sys.getenv("WRDS_PASSWORD")
  
  if (wrds_username == "" || wrds_password == "") {
    stop("WRDS credentials not found. Please set WRDS_USERNAME and WRDS_PASSWORD.")
  }
  
  list(
    username = wrds_username,
    password = wrds_password
  )
}
