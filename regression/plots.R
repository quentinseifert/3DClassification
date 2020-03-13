# skript containing the plots for the explanation of the intuition 

#1

setwd("..Buche")
Bu1 <- read.table("Bu1.xyz", header = FALSE)

setwd("..Fichte")
Fi1 <- read.table("Jerome_Fi5_AEW29.xyz", header = FALSE)


set.seed(100)
abc<-sample(1:length(Bu1$V1),10000) 
plot(Bu1$V1[abc],Bu1$V2[abc], ylim = c(-8,8), xlim = c(-12, 6),col=4
     , xlab = "X Values", ylab = "Y Values", main ="Top down view of a spruce and a beech")

abc<-sample(1:length(Fi1$V1),5000) 
points(Fi1$V1[abc], Fi1$V2[abc], col=2)

legend(x = -10, y = -5, col=c("blue","red"),  legend = c("beech","spruce"), pch = 1)

############################################

#2

setwd("..Buche")
Bu2 <- read.table("Jerome_Bu1_HEW29.xyz", header = FALSE)

setwd("..Fichte")
Fi2 <- read.table("Jerome_Fi57_AEW47.xyz", header = FALSE)

set.seed(100)
abc<-sample(1:length(Bu2$V1),10000) 
plot(Bu2$V1[abc],Bu2$V2[abc], ylim = c(-5,10), xlim = c(-2, 12),col=4
     , xlab = "X Values", ylab = "Y Values", main ="Top down view of a spruce and a beech")

abc<-sample(1:length(Fi2$V1), 5000) 
points(Fi2$V1[abc], Fi2$V2[abc], col=2)

legend(x = -1, y = 10, col=c("blue","red"),  legend = c("beech","spruce"), pch = 1)



####################################################################

#3

setwd("..Buche")
Bu2 <- read.table("Bu6.xyz", header = FALSE)

setwd("..Fichte")
Fi2 <- read.table("Jerome_Fi74_AEW29.xyz", header = FALSE)

set.seed(100)
abc<-sample(1:length(Bu2$V1),10000) 
plot(Bu2$V1[abc],Bu2$V2[abc], ylim = c(-8,10), xlim = c(-10, 3),col=4
     , xlab = "X Values", ylab = "Y Values", main ="Top down view of a spruce and a beech")

abc<-sample(1:length(Fi2$V1), 5000) 
points(Fi2$V1[abc], Fi2$V2[abc], col=2)

legend(x = 0, y = 10, col=c("blue","red"),  legend = c("beech","spruce"), pch = 1)

