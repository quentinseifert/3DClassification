ahorn <- diameter(dataahorn2)
roteiche <- diameter(dataroteiche2)
eiche <- diameter(dataeiche2)
linde <- diameter(datalinde2)
hainbuche <- diameter(datahainbuche2)
esche <- diameter(dataesche2)
buche <- diameter(databuche2)

douglasie <- diameter(datadouglasie2)
fichte <- diameter(datafichte2)
kiefer <- diameter(datakiefer2)


#################

dummy<-rep(1, times = length(ahorn[, 1])); ahorn <- cbind(dummy, ahorn)
dummy<-rep(1, times = length(roteiche[, 1])); roteiche <-cbind(dummy, roteiche)
dummy<-rep(1, times = length(eiche[, 1])); eiche <-cbind(dummy, eiche)
dummy<-rep(1, times = length(linde[, 1])); linde <-cbind(dummy, linde)
dummy<-rep(1, times = length(hainbuche[, 1])); hainbuche <-cbind(dummy, hainbuche)
dummy<-rep(1, times = length(esche[, 1])); esche <-cbind(dummy, esche)
dummy<-rep(1, times = length(buche[, 1])); buche <-cbind(dummy, buche)

dummy<-rep(0, times = length(kiefer[, 1])); kiefer <-cbind(dummy, kiefer)
dummy<-rep(0, times = length(fichte[, 1])); fichte <-cbind(dummy, fichte)
dummy<-rep(0, times = length(douglasie[, 1])); douglasie <-cbind(dummy, douglasie)


###### trunkless set

Z <- rbind(kiefer, fichte, douglasie, ahorn, roteiche, eiche, linde, hainbuche, esche, buche)

########################################################################################
cor(Z[,2], Z[,3])
Z <- Z[,-2]

#########################################################################################
#glm with full set

myglm1.1<-glm(dummy ~ horizontal, data = Z, family = binomial(link
                                                              ="logit"))
summary(myglm1.1)

# classifying 
# split into test and training set

set.seed(12)

a <- sample(dim(Z)[1], round(dim(Z)[1]/4))

trainingset <- Z[-a,]

testingset <- Z[a,]

#logit model based on the trainingset

myglm1.2<-glm(dummy ~ horizontal, data = trainingset, family = binomial(link
                                                                        ="logit"))


# using the trained myglm1.2 model to test it with testingset

prediction <- rep(NA, times = dim(testingset)[1])

for (i in 1:length(prediction)) {
  
  prediction[i] <- predict(myglm1.2, type = "response", newdata = 
                             data.frame("horizontal" = testingset[i,2]))
}


#correct classification causes a 1 
#incorrect causes a 0

classification <- rep(0, times= length(prediction))

for (i in 1: length(prediction)) {
  
  if (prediction[i]>0.5 & testingset[i,1]==1) {
    classification[i] <- 1
  } 
  
  if (prediction[i] <= 0.5 & testingset[i,1]==0) {
    classification[i] <- 1
  } 
  
}

mean(classification)
mean(testingset[,1])
