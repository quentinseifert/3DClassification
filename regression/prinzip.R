# Skript zur Erläuterung des Vorgehens 

#1) Einlesen 

setwd("../Buche")
Bu1 <- read.table("Bu1.xyz", header = FALSE)

#2) Ein erster Plot

abc<-sample(1:length(Bu1$V1),10000)
plot(Bu1$V1[abc],Bu1$V2[abc])

#3) Bestimmung der indizes der aeußeren Punkte

minpos1 <- which.min(Bu1$V1)
maxpos1 <- which.max(Bu1$V1)
minpos2 <- which.min(Bu1$V2)
maxpos2 <- which.max(Bu1$V2)

#4) Zusammenfassung der tatsaechlichen Werte mithilfe der Indizes
#   in zwei Vektoren

e<-c(Bu1$V1[minpos1],Bu1$V1[maxpos1],Bu1$V1[minpos2],Bu1$V1[maxpos2])
f<-c(Bu1$V2[minpos1],Bu1$V2[maxpos1],Bu1$V2[minpos2],Bu1$V2[maxpos2])

#5) Veranschaulichung, dass es die äußeren Punkte sind mit Hilfe eines
#   weiteren PLots

plot(e,f)


#6) Zusammenfassung der tatsaechlichen Werte in einer Matrix 
#   um sie spaeter nur an eine Funktion weiter geben zu 
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









