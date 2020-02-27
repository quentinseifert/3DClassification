#Funktionen für Durchmesser der Baumkrone


# Vorbereitung für Durchmesser von der Baumkrone 


satzprep1 <- function(x) {
  
  # hier das minimum und maximum von der x achse (V1) also 
  # der horizontale schnitt
  
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
  
  # hier  das minimum und maximum von der y achse (V2), also 
  # der vertikale schnitt
  
  
  minpos1 <- which.min(x$V1)
  maxpos1 <- which.max(x$V1)
  minpos2 <- which.min(x$V2)
  maxpos2 <- which.max(x$V2)
  
  e<-c(x$V1[minpos1],x$V1[maxpos1],x$V1[minpos2],x$V1[maxpos2])
  f<-c(x$V2[minpos1],x$V2[maxpos1],x$V2[minpos2],x$V2[maxpos2])
  
  krone2 <- matrix(c(e[3],e[4],f[3],f[4]), ncol = 2, byrow = TRUE)
  
  return(krone2)
  
}


# satz des pythagoras

##### wichtig

# Die x Koordinaten der zwei Punkte deren Abstand gemessen werden soll
# müssen in der zu einlesenden 2x2 matrix an position (1,1) und (1,2)
# stehen. Die dazu gehörigen y Koordinaten an position (2,1) und (2,2)

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
  
  # erster zwischenspeicher für die 2 koordinaten (4 werte) horizontal
  
  zwischenspeicher1 <- vector(mode = "list", length = length(dataset))
  
  
  for (i in 1:length(dataset)) {
    
    zwischenspeicher1[[i]]<-satzprep1(dataset[[i]])
    
  }
  
  
  # erster zwischenspeicher für die 2 koordinaten (4 werte) vertical
  
  
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







