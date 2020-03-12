# loading in the data

# this file contains the code for reading in the data
# it is expected that you have the data in your working directory
# in order to create a list which includes the datafile names:
# at first one has to setwd() to the MAIN file which includes the
# SUB files ->  setwd(".../Laubbäume"). In the list.files command
# one has to specifie which sub file he wants to adress -> "Hainbuche"
# for example. After one has listet every datafile name in a list, 
# one hase to change the working directory again. This time
# NOT to the main file but to the SUB file -> setwd("../Laubbäume/Hainbuche")

#this file automattically cuts the trunk and deletes the third column
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
## leaf trees : 


################################################################################
# linde / lime tree

setwd("")
B <- list.files(path = "")
datalinde <- vector(mode = "list", length = length(B))

setwd('')


# daten in datalinde liste einlesen 

for (i in 1:length(B)) {
  
  datalinde[[i]]<-read.table(B[i], header = FALSE)
  
}

datalinde2<- list(NA)
for (i in 1:length(datalinde)) {
  
  datalinde2[[i]] <- datalinde[[i]][-which(datalinde[[i]]$V3<(min(datalinde[[i]]$V3))+7),]
  datalinde2[[i]] <- datalinde2[[i]][,c(1,2)]
}

################################################################################
#roteiche /  red oak



setwd('')
A <- list.files(path = "")
dataroteiche <- vector(mode = "list", length = length(A))


setwd('')


for (i in 1:39) {
  
  dataroteiche[[i]]<-read.table(A[i], header = FALSE, sep = ",")
  
}

dataroteiche [[40]] <- read.table(A[40], header = FALSE, sep = "")

for (i in 1:(length(A)-40)) {
  
  dataroteiche[[(40+i)]]<-read.table(A[(40+i)], header = FALSE, sep = ",")
  
}

# datensätze in dataroteich zuschneiden (die letzten spalten müssen weg)

for (i in 1: length(A)) {
  
  dataroteiche[[i]] <- dataroteiche[[i]][,1:3]
}

dataroteiche2<- list(NA)
for (i in 1:length(dataroteiche)) {
  
  dataroteiche2[[i]] <- dataroteiche[[i]][-which(dataroteiche[[i]]$V3<(min(dataroteiche[[i]]$V3))+7),]
  dataroteiche2[[i]] <- dataroteiche2[[i]][,c(1,2)]
  
}



################################################################################
#ahorn

setwd("")
C <- list.files(path = "")
dataahorn <- vector(mode = "list", length = length(C))
setwd('')


for (i in 1:length(C)) {
  
  dataahorn[[i]]<-read.table(C[i], header = FALSE)
  
}

dataahorn2<- list(NA)
for (i in 1:length(dataahorn)) {
  
  dataahorn2[[i]] <- dataahorn[[i]][-which(dataahorn[[i]]$V3<(min(dataahorn[[i]]$V3))+7),]
  dataahorn2[[i]] <- dataahorn2[[i]][,c(1,2)]
  
}


################################################################################
# buche


setwd("")
D <- list.files(path = "")
databuche <- vector(mode = "list", length = length(D))
setwd('')


databuche [[1]] <- read.table(D[1], header = FALSE)
databuche [[2]] <- read.table(D[2], header = FALSE)
databuche [[3]] <- read.table(D[3], header = FALSE)
databuche [[4]] <- read.table(D[4], header = FALSE, sep = ",")
databuche [[5]] <- read.table(D[5], header = FALSE, sep = " ")
databuche [[6]] <- read.table(D[6], header = FALSE)

for (i in 1:53) {
  
  databuche[[(6+i)]]<-read.table(D[(6+i)], header = FALSE)
  
}


databuche [[60]] <- read.table(D[60], header = FALSE, sep = ",")


for (i in 1 : (length(D)-60)) {
  
  databuche[[(60+i)]]<-read.table(D[(60+i)], header = FALSE)
  
}

databuche2<- list(NA)
for (i in 1:length(databuche)) {
  
  databuche2[[i]] <- databuche[[i]][-which(databuche[[i]]$V3<(min(databuche[[i]]$V3))+7),]
  databuche2[[i]] <- databuche2[[i]][,c(1,2)]
  
}







################################################################################
# eiche


setwd('')
E <- list.files(path = "")
dataeiche <- vector(mode = "list", length = length(E))
setwd('')

for (i in 1:length(E)) {
  
  dataeiche[[i]]<-read.table(E[i], header = FALSE, sep = ";")
  
}

for (i in 1: length(E)) {
  
  dataeiche[[i]] <- dataeiche[[i]][,1:3]
}

dataeiche2<- list(NA)
for (i in 1:length(dataeiche)) {
  
  dataeiche2[[i]] <- dataeiche[[i]][-which(dataeiche[[i]]$V3<(min(dataeiche[[i]]$V3))+7),]
  dataeiche2[[i]] <- dataeiche2[[i]][,c(1,2)]
  
}





################################################################################
# Esche


setwd('')
G <- list.files(path = "")
dataesche <- vector(mode = "list", length = length(G))
setwd('')

for (i in 1:length(G)) {
  
  dataesche[[i]]<-read.table(G[i], header = FALSE)
  
}



dataesche2<- list(NA)
for (i in 1:length(dataesche)) {
  
  dataesche2[[i]] <- dataesche[[i]][-which(dataesche[[i]]$V3<(min(dataesche[[i]]$V3))+7),]
  dataesche2[[i]] <- dataesche2[[i]][,c(1,2)]
  
}


################################################################################
# Hainbuche


setwd('')
H <- list.files(path = "")
datahainbuche <- vector(mode = "list", length = length(H))
setwd('')

for (i in 1:length(H)) {
  
  datahainbuche[[i]]<-read.table(H[i], header = FALSE)
  
}

datahainbuche2<- list(NA)
for (i in 1:length(datahainbuche)) {
  
  datahainbuche2[[i]] <- datahainbuche[[i]][-which(datahainbuche[[i]]$V3<(min(datahainbuche[[i]]$V3))+7),]
  datahainbuche2[[i]] <- datahainbuche2[[i]][,c(1,2)]
  
}




################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
## decidious trees :

################################################################################
## douglasie :



setwd('')
I <- list.files(path = "")
datadouglasie <- vector(mode = "list", length = length(I))
setwd('')

for (i in 1:length(I)) {
  
  datadouglasie[[i]]<-read.table(I[i], header = FALSE)
  
}


datadouglasie2<- list(NA)
for (i in 1:length(datadouglasie)) {
  
  datadouglasie2[[i]] <- datadouglasie[[i]][-which(datadouglasie[[i]]$V3<(min(datadouglasie[[i]]$V3))+4),]
  datadouglasie2[[i]] <- datadouglasie2[[i]][,c(1,2)]
  
}



################################################################################
## fichte :



setwd('')
J <- list.files(path = "")
datafichte <- vector(mode = "list", length = length(J))
setwd('')

for (i in 1:length(J)) {
  
  datafichte[[i]]<-read.table(J[i], header = FALSE)
  
}


datafichte2<- list(NA)
for (i in 1:length(datafichte)) {
  
  datafichte2[[i]] <- datafichte[[i]][-which(datafichte[[i]]$V3<(min(datafichte[[i]]$V3))+4),]
  datafichte2[[i]] <- datafichte2[[i]][,c(1,2)]
  
}



################################################################################
## kiefer :


setwd('')
K <- list.files(path = "")
datakiefer <- vector(mode = "list", length = length(K))
setwd('')

for (i in 1:length(K)) {
  
  datakiefer[[i]]<-read.table(K[i], header = FALSE)
  
}


datakiefer2<- list(NA)
for (i in 1:length(datakiefer)) {
  
  datakiefer2[[i]] <- datakiefer[[i]][-which(datakiefer[[i]]$V3<(min(datakiefer[[i]]$V3))+8),]
  datakiefer2[[i]] <- datakiefer2[[i]][,c(1,2)]
  
}

