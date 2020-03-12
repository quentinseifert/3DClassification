# function for calculating the maximal distance between two points 

d <- function(dat) {
  
  speicher <- rep(NA, times = length(dat))
  
  for (i in 1:length(dat)) {

random <- sample(1:dim(dat[[i]])[1], dim(dat[[i]])[1]*0.1)
  
zwischenspeicher <- dat[[i]][random,]

speicher[i] <- max(dist(zwischenspeicher))
  }
  
  return(speicher)
}


dlinde <- d(datalinde2)
droteiche <- d(dataroteiche2)
dbuche <- d(databuche2)
dahorn <- d(dataahorn2)
deiche <- d(dataeiche2)
desche <- d(dataesche2)
dhainbuche <- d(datahainbuche2)


d2 <- function(dat) {
  
  speicher <- rep(NA, times = length(dat))
  
  for (i in 1:length(dat)) {
    
    random <- sample(1:dim(dat[[i]])[1], 10000)
    
    zwischenspeicher <- dat[[i]][random,]
    
    speicher[i] <- max(dist(zwischenspeicher))
  }
  
  return(speicher)
}

ddouglasie <- d2(datadouglasie2)
dfichte <- d2(datafichte2)
dkiefer <- d2(datakiefer2)




