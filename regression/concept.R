# Skript zur Erläuterung des Vorgehens 

#1) Einlesen 

setwd("")
Bu1 <- read.table("Bu1.xyz", header = FALSE)

setwd("")
Ki1 <- read.table("Jerome_Ki7_SEW4.xyz", header = FALSE)

setwd("")
D1 <- read.table("Tree19_Interior.xyz", header = FALSE)


##################################### Buche
#  Plots (in order to reduce the complexity of the plot, 10000 data points
#         of the point clouds are selected)

set.seed(100)
abc<-sample(1:length(Bu1$V1),10000) 
plot(Bu1$V1[abc],Bu1$V2[abc])

#1) Bestimmung der indizes der aeußeren Punkte

minpos1 <- which.min(Bu1$V1)
maxpos1 <- which.max(Bu1$V1)
minpos2 <- which.min(Bu1$V2)
maxpos2 <- which.max(Bu1$V2)

#2) Zusammenfassung der tatsaechlichen Werte mithilfe der Indizes
#   in zwei Vektoren

e<-c(Bu1$V1[minpos1],Bu1$V1[maxpos1],Bu1$V1[minpos2],Bu1$V1[maxpos2])
f<-c(Bu1$V2[minpos1],Bu1$V2[maxpos1],Bu1$V2[minpos2],Bu1$V2[maxpos2])

#3) Veranschaulichung, dass es die äußeren Punkte sind mit Hilfe eines
#   weiteren PLots

points(e,f, col = 2)
plot(e,f)

######################################douglasie
#  Plots (in order to reduce the complexity of the plot, 10000 data points
#         of the point clouds are selected)

set.seed(50)
abc<-sample(1:length(D1$V1),10000) 
plot(D1$V1[abc], D1$V2[abc])

#1) Bestimmung der indizes der aeußeren Punkte
minpos1 <- which.min(D1$V1)
maxpos1 <- which.max(D1$V1)
minpos2 <- which.min(D1$V2)
maxpos2 <- which.max(D1$V2)

#2) Zusammenfassung der tatsaechlichen Werte mithilfe der Indizes
#   in zwei Vektoren

e<-c(D1$V1[minpos1],D1$V1[maxpos1],D1$V1[minpos2],D1$V1[maxpos2])
f<-c(D1$V2[minpos1],D1$V2[maxpos1],D1$V2[minpos2],D1$V2[maxpos2])

#3) Veranschaulichung, dass es die äußeren Punkte sind mit Hilfe eines
#   weiteren PLots

plot(e,f)
points(e,f, col = 2)


######################################kiefer

# this trees reveals the following issue
# the tree might be skewed in a way that 
# the trunk appears to be the most outer part
# and not the treetop
abc<-sample(1:length(Ki1$V1),10000) 
plot(Ki1$V2[abc], Ki1$V3[abc])

# cut the trunk 
Ki1 <- Ki1[-which(Ki1$V3<1),]
abc<-sample(1:length(Ki1$V1),10000) 
plot(Ki1$V2[abc], Ki1$V3[abc])




#############################################
# After this visualization
# using again the Buche example
minpos1 <- which.min(Bu1$V1)
maxpos1 <- which.max(Bu1$V1)
minpos2 <- which.min(Bu1$V2)
maxpos2 <- which.max(Bu1$V2)

#2) Zusammenfassung der tatsaechlichen Werte mithilfe der Indizes
#   in zwei Vektoren

e<-c(Bu1$V1[minpos1],Bu1$V1[maxpos1],Bu1$V1[minpos2],Bu1$V1[maxpos2])
f<-c(Bu1$V2[minpos1],Bu1$V2[maxpos1],Bu1$V2[minpos2],Bu1$V2[maxpos2])

#6) Zusammenfassung der tatsaechlichen Werte in einer Matrix 
#   um sie spaeter an eine Funktion weiter geben zu 
#   koennen.

# krone1 als horizontaler schnitt
krone1 <- matrix(c(e[1],e[2],f[1],f[2]), ncol = 2, byrow = TRUE)

# krone2 als vertikaler schnitt entlang der y achse
krone2 <- matrix(c(e[3],e[4],f[3],f[4]), ncol = 2, byrow = TRUE)

#7) Oder direkt mit den funktionen satzprep
#   diese lesen einen datensatz ein welcher drei spalten umfasst.
#   Diese Funktion ermittelt dann die jeweiligen maximalen und 
#   und minimalen werte und gibt diese in matrix form aus.

a<-satzprep1(Bu1)
b<-satzprep2(Bu1)

#8) satz des Pythagoras welche die vorbereitete matrix uebernimmt.

satz(a)
satz(b)









