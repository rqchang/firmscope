
# ====================================================== #
# Calculate Book Equity value ####
# ====================================================== #


calculate_be <- function(mmdata, freq = 'quarterly'){
  mdata <- copy(mmdata)
  
  # Calculate Book Equity (BE) following Fama and French
  # ---------------------------------------------------- #
  # Input: 
  #   cdata - data table. Required columns: 'seq', 'ceq', 'pstk', 'at', 'lt', 'mib'  
  #   freq  - character. annual or quarterly 
  #
  # Output: 
  #   be - series
  # ---------------------------------------------------- #
  
  if(freq == 'annual'){
    # Create BE annual (more populated than quartely)
    # Shareholder Equity
    mdata[,'se':=seq]
    # Use Common Equity (ceq) + Preferred Stock (pstk) if SEQ is missing:
    mdata[is.na(se),se:=ceq+pstk]
    # Uses Total Assets (at) - Liabilities (lt) + Minority Interest (mib, if available), if others are missing
    mdata[is.na(se)&!is.na(mib), se:=at-lt+mib]
    mdata[is.na(se), se:=at-lt]
    
    # Preferred Stock
    # Preferred Stock (Redemption Value)
    mdata[,'ps':=pstkrv]
    # Uses Preferred Stock (Liquidating Value (pstkl)) if Preferred Stock (Redemption Value) is missing
    mdata[is.na(ps),'ps':=pstkl]
    # Uses Preferred Stock (Carrying Value (pstk)) if others are missing
    mdata[is.na(ps),'ps':=pstk]
    
    # Deferred Taxes
    # Uses Deferred Taxes and Investment Tax Credit (txditc)
    mdata[,'dt':=txditc]
    # Uses Deferred Taxes and Investment Tax Credit(txdb) + Investment Tax Credit (Balance Sheet) (itcb) if txditc is missing
    mdata[is.na(dt),'dt':=txdb+itcb]
    
    # Book Equity
    # Book Equity (BE) = Share Equity (se) - Prefered Stocks (ps) + Deferred Taxes (dt)
    ## shareholder equity must be available, otherwise BE is missing
    ## preferred stock must be available, otherwise BE is missing
    ## add deferred taxes if available
    ## subtract postretirement benefit assets if available
    mdata[,'be':=se-ps]
    mdata[!is.na(dt)&fyear<=1993,be:=be+dt]
    return(mdata[,.(be)])
  }
 
  else{
    # Shareholder Equity
    mdata[,'se':=seqq]
    # Uses Common Equity (ceq) + Preferred Stock (pstk) if SEQ is missing:
    mdata[is.na(se),se:=ceqq+pstkq]
    # Uses Total Assets (at) - Liabilities (lt) + Minority Interest (mib, if available), if others are missing
    mdata[is.na(se)&!is.na(mibq), se:=atq-ltq+mibq]
    mdata[is.na(se), se:=atq-ltq]
    
    # Preferred Stock
    # Preferred Stock (Redemption Value)
    #mdata[,'ps':=pstkrv]
    # Uses Preferred Stock (Liquidating Value (pstkl)) if Preferred Stock (Redemption Value) is missing
    #mdata[is.na(ps),'ps':=pstklyy]
    # Uses Preferred Stock (Carrying Value (pstk)) if others are missing
    #mdata[is.na(ps),'ps':=pstk]
    mdata[,'ps':=pstkq]
    
    # Deferred Taxes
    # Uses Deferred Taxesand Investment Tax Credit (txditc)
    mdata[,'dt':=txditcq]
    # Uses Deferred Taxes and Investment Tax Credit(txdb) + Investment Tax Credit (Balance Sheet) (itcb) if txditc is missing
    # mdata[is.na(dt),'dt':=txdb+itcbyy]
    
    # Book Equity
    # Book Equity (BE) = Share Equity (se) - Prefered Stocks (ps) + Deferred Taxes (dt)
    ## shareholder equity must be available, otherwise BE is missing
    ## preferred stock must be available, otherwise BE is missing
    ## add deferred taxes if available
    ## subtract postretirement benefit assets if available
    mdata[,'beq':=se-ps]
    mdata[!is.na(dt)&fyearq<=1993,beq:=beq+dt]
    return(mdata[,.(beq)])
    
  }
  }


# ====================================================== #
# Variables from cash analyses ####
# ====================================================== #

# Calculate Net Operating Profits
calculate_nop <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'oancf', 'iinv')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    nop = adata[,oancf + na.fill(iinv, 0)]
    nop[is.na(adata$chech)] = NA
    
    return(nop)} else{
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'oancfq', 'iinvq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        nop = adata[,oancfq + na.fill(iinvq, 0)]
        nop[is.na(adata$chechq)] = NA
        
        return(nop)}
      
    }
}

# Calculate Net Debt Issuance
calculate_stndi <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'dlcch')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    stndi <- adata[,na.fill(dlcch, 0)]
    stndi[is.na(adata$chech)] <-  NA
    return(stndi)} else{
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'dlcchq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        stndi <- adata[,na.fill(dlcchq, 0)]
        stndi[is.na(adata$chechq)] <-  NA
        return(stndi)}}
  
  
}

calculate_ndi <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'dlcch')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    ndi <- adata[,na.fill(dltis, 0) - na.fill(dltr, 0)]
    ndi[is.na(adata$chech)] <-  NA
    return(ndi)} else{
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'dlcchq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        ndi <- adata[,na.fill(dltisq, 0) - na.fill(dltrq, 0)]
        ndi[is.na(adata$chechq)] <-  NA
        return(ndi)}}
}

# Calculate Total Net Payout
# Net equity repurchase
calculate_ner <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'prstkc', 'sstk')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    ner <- adata[,na.fill(prstkc, 0) - na.fill(sstk, 0)]
    ner[is.na(adata$chech)] <-  NA
    return(ner)}else{
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'prstkcq', 'sstkq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        ner <- adata[,na.fill(prstkcq, 0) - na.fill(sstkq, 0)]
        ner[is.na(adata$chechq)] <-  NA
        return(ner)}
    }
}

# Cash dividends
calculate_cdv <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'dv')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    cdv <- adata[,na.fill(dv, 0)]
    cdv[is.na(adata$chech)] <-  NA
    return(cdv)} else{
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'dvq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        cdv <- adata[,na.fill(dvq, 0)]
        cdv[is.na(adata$chechq)] <-  NA
        return(cdv)}
    }
}

# Real investments
# Capital investment
calculate_kinv <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'capx', 'sppe')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    kinv <- adata[,capx - na.fill(sppe, 0)]
    kinv[is.na(adata$chech)] <-  NA
    return(kinv)} else {
      if(freq == "quarterly"){
        required_columns <- c('chechq', 'capxq', 'sppeq')
        if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
        
        kinv <- adata[,capxq - na.fill(sppeq, 0)]
        kinv[is.na(adata$chechq)] <-  NA
        return(kinv)}
    }
}

# Total acquisitions
calculate_taqc <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'aqc')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    taqc <- adata[,na.fill(aqc, 0)]
    taqc[is.na(adata$chech)] <-  NA
    return(taqc)
  }else{
    if(freq == "quarterly"){
      required_columns <- c('chechq', 'aqcq')
      if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
      
      taqc <- adata[,na.fill(aqcq, 0)]
      taqc[is.na(adata$chechq)] <-  NA
      return(taqc)
    }
  }
}

# Total acquisitions with SDC values
calculate_taqc_SDC <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'aqcq_SDC')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    taqc <- adata[,na.fill(aqcq_SDC, 0)]
    taqc[is.na(adata$chech)] <-  NA
    return(taqc)
  } else {
    if(freq == "quarterly"){
      required_columns <- c('chechq', 'aqcq_SDC')
      if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
      
      taqc <- adata[,na.fill(aqcq_SDC, 0)]
      taqc[is.na(adata$chechq)] <-  NA
      return(taqc)
    }
  }
}

# Total intangible
calculate_iinv <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'xsga', 'xrd', 'rdip')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    sga <- adata[,xsga - na.fill(xrd, 0) + na.fill(rdip, 0)]
    to_change <- which(adata$xsga<adata$xrd & adata$xsga<adata$cogs)
    sga[to_change] = adata[to_change, xsga]
    
    sga[is.na(adata$chech)] <-  NA
    
    iinv <- adata[,ifelse(is.na(xrd), 0, xrd)] + 0.3*sga
    
    return(iinv)
  }else{
    if(freq == "quarterly"){
      required_columns <- c('chechq', 'xsgaq', 'xrdq', 'rdipq')
      if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
      
      sga <- adata[,xsgaq - na.fill(xrdq, 0) + na.fill(rdipq, 0)]
      to_change <- which(adata$xsgaq<adata$xrdq & adata$xsgaq<adata$cogsq)
      sga[to_change] = adata[to_change, xsgaq]
      
      sga[is.na(adata$chechq)] <-  NA
      
      iinv <- adata[,ifelse(is.na(xrdq), 0, xrdq)] + 0.3*sga
      
      return(iinv)
    }
  }}

# Total financial investment 
calculate_finv <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'ivch', 'siv', 'ivstch')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    finv <- adata[,na.fill(ivch, 0) - na.fill(siv, 0) - na.fill(ivstch, 0)]
    finv[is.na(adata$chech)] <-  NA
    
    return(finv)
  }else{
    if(freq == "quarterly"){
      required_columns <- c('chechq', 'ivchq', 'sivq', 'ivstchq')
      if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
      
      finv <- adata[,na.fill(ivchq, 0) - na.fill(sivq, 0) - na.fill(ivstchq, 0)]
      finv[is.na(adata$chechq)] <-  NA
      
      return(finv)
    }
  }}

# Other expenses  
calculate_oex <- function(adata, freq = "annual"){
  if(freq == "annual"){
    required_columns <- c('chech', 'ivaco', 'fiao', 'txbcof', 'exre')
    if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
    
    oex <- adata[,-na.fill(ivaco,0) - na.fill(fiao,0) - na.fill(txbcof,0) - na.fill(exre,0)]
    oex[is.na(adata$chech)] <-  NA
    
    return(oex)
  }else{
    if(freq == "quarterly"){
      required_columns <- c('chechq', 'ivacoq', 'fiaoq', 'txbcofq', 'exreq')
      if(!all(required_columns%in%colnames(adata))) stop('Missing columns in data.')
      
      oex <- adata[,-na.fill(ivacoq,0) - na.fill(fiaoq,0) - na.fill(txbcofq,0) - na.fill(exreq,0)]
      oex[is.na(adata$chechq)] <-  NA
      
      return(oex)
    }
  }}
