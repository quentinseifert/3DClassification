#skript containing the functions to calculate the diameter

# preperation


satzprep1 <- function(x) {
  
  # minimum and maximum of the X values (V1), so the horizontal cut
  
  minpos1 <- which.min(x$V1)
  maxpos1 <- which.max(x$V1)
  minpos2 <- which.min(x$V2)
  maxpos2 <- which.max(x$V2)
  
  e<-c(x$V1[minpos1],x$V1[maxpos1],x$V1[minpos2],x$V1[maxpos2])
  f<-c(x$V2[minpos1],x$V2[maxpos1],x$V2[minpos2],x$V2[maxpos2])
  
  krone1 <- matrix(c(e[1],e[2],f[1],f[2]), ncol = 2, byrow = TRUE)
  
  return(krone1)
  
}

satzprep2 <- function(x) {
  
  # minimum and maximum of the Y values (V2), so the vertical cut
  
  minpos1 <- which.min(x$V1)
  maxpos1 <- which.max(x$V1)
  minpos2 <- which.min(x$V2)
  maxpos2 <- which.max(x$V2)
  
  e<-c(x$V1[minpos1],x$V1[maxpos1],x$V1[minpos2],x$V1[maxpos2])
  f<-c(x$V2[minpos1],x$V2[maxpos1],x$V2[minpos2],x$V2[maxpos2])
  
  krone2 <- matrix(c(e[3],e[4],f[3],f[4]), ncol = 2, byrow = TRUE)
  
  return(krone2)
  
}


#theorem of pythagoras
# The x coordinates of the two points, where the distance wants to be calculated
# need to be at position (1,1) and (1,2) in the used 2x2 matrix 
# the corresponding y coordinates have to be at position (2,1) and (2,2)

satz <- function( matr ) {
  
  if (dim(matr)[1]==2 & dim(matr)[2]==2) {
    
    laenge <-(matr[1,1]-matr[1,2])^2+(matr[2,1]-matr[2,2])^2
    
  } else {
    
    print("Wrong dimensions")
  }
  
  return(sqrt(laenge))
  
} 


##################################################################

diameter <- function(dataset) {
  
  # first temporary storage of the 2 coordinates of the horizontal cut (4 values)
  
  zwischenspeicher1 <- vector(mode = "list", length = length(dataset))
  
  
  for (i in 1:length(dataset)) {
    
    zwischenspeicher1[[i]]<-satzprep1(dataset[[i]])
    
  }
  
  # second temporary storage of the 2 coordinates of the vertical cut (4 values)
  
  zwischenspeicher2 <- vector(mode = "list", length = length(dataset))
  
  
  for (i in 1:length(dataset)) {
    
    zwischenspeicher2[[i]]<-satzprep2(dataset[[i]])
    
  }
  
  
  ##################################################
  
  horizontal <- rep(NA, times = length(dataset))
  
  for (i in 1:length(dataset)) {
    
    horizontal[i] <- satz(zwischenspeicher1[[i]])
  }
  
  
  vertical <- rep(NA, times = length(dataset))
  
  for (i in 1:length(dataset)) {
    vertical[i] <- satz(zwischenspeicher2[[i]])
    
  }
  
  #############################################
  
  return(data.frame(vertical, horizontal))
}