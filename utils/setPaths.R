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
           DATADIR <- "D:/Dropbox/project/firm_scope/data"
           RAWDIR <- "D:/Dropbox/project/firm_scope/data/raw/"
           TEMPDIR <- "D:/Dropbox/project/firm_scope/data/temp/"
           PROCDIR <- "D:/Dropbox/project/firm_scope/data/processed/"
           OUTDIR <- "D:/Dropbox/project/firm_scope/outputs/"
           PLOTSDIR <- "D:/Dropbox/project/firm_scope/outputs/plots/"
           TABLESDIR <- "D:/Dropbox/project/firm_scope/outputs/tables/"
         },
         "ziqili" = {
           DATADIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/data"
           RAWDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/data/raw/"
           TEMPDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/data/temp/"
           PROCDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/data/processed/"
           OUTDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/outputs/"
           PLOTSDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/outputs/plots/"
           TABLESDIR <- "/Users/ziqili/Dropbox (Personal)/projects/firm_scope/outputs/tables/"
         }
         )
}
