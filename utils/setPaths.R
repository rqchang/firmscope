# Default Sloan Cluster
if(Sys.info()['sysname']=="Linux"){
  DATADIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/"
  RAWDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/raw/"
  TEMPDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/temp/"
  PROCDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/data/processed/"
  OUTDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/"
  PLOTSDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/plots/"
  TABLESDIR <- "/nfs/sloanlab003/projects/systematic_credit_proj/outputs/tables/"
}else{
  switch(Sys.info()["user"],
         "chang.2590" = {
           DATADIR <- "D:/Dropbox/project/fund_contagion/data"
           RAWDIR <- "D:/Dropbox/project/fund_contagion/data/raw/"
           TEMPDIR <- "D:/Dropbox/project/fund_contagion/data/temp/"
           PROCDIR <- "D:/Dropbox/project/fund_contagion/data/processed/"
           OUTDIR <- "D:/Dropbox/project/fund_contagion/outputs/"
           PLOTSDIR <- "D:/Dropbox/project/fund_contagion/outputs/plots/"
           TABLESDIR <- "D:/Dropbox/project/fund_contagion/outputs/tables/"
         },
         "ziqili" = {
           DATADIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/data"
           RAWDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/data/raw/"
           TEMPDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/data/temp/"
           PROCDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/data/processed/"
           OUTDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/outputs/"
           PLOTSDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/outputs/plots/"
           TABLESDIR <- "/Users/ziqili/Dropbox (Personal)/projects/fund_contagion/outputs/tables/"
         }
         )
}
