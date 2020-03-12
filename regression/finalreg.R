#skript containing the assembly of the data and the regression


dummy<-rep(1, times = length(dahorn)); ahorn <- cbind(dummy, dahorn)
dummy<-rep(1, times = length(droteiche)); roteiche <-cbind(dummy, droteiche)
dummy<-rep(1, times = length(deiche)); eiche <-cbind(dummy, deiche)
dummy<-rep(1, times = length(dlinde)); linde <-cbind(dummy, dlinde)
dummy<-rep(1, times = length(dhainbuche)); hainbuche <-cbind(dummy, dhainbuche)
dummy<-rep(1, times = length(desche)); esche <-cbind(dummy, desche)
dummy<-rep(1, times = length(dbuche)); buche <-cbind(dummy, dbuche)

dummy<-rep(0, times = length(dkiefer)); kiefer <-cbind(dummy, dkiefer)
dummy<-rep(0, times = length(dfichte)); fichte <-cbind(dummy, dfichte)
dummy<-rep(0, times = length(ddouglasie)); douglasie <-cbind(dummy, ddouglasie)


###### data set

Z <- rbind(kiefer, fichte, douglasie, ahorn, roteiche, eiche, linde, hainbuche, esche, buche)
colnames(Z) <- c("species","diameter")
Z <- as.data.frame(Z)
#########################################################################################
#glm-logit model

myglm1.1<-glm(species ~ diameter, data = Z, family = binomial(link
                                                              ="logit"))
summary(myglm1.1)

# classifying 
# split into test and training set

set.seed(12)

a <- sample(dim(Z)[1], round(dim(Z)[1]/4))

trainingset <- Z[-a,]

testingset <- Z[a,]

#logit model based on the trainingset

myglm1.2<-glm(species ~ diameter, data = trainingset, family = binomial(link
                                                                        ="logit"))

# using the trained myglm1.2 model to test it with testingset

prediction <- rep(NA, times = dim(testingset)[1])

for (i in 1:length(prediction)) {
  
  prediction[i] <- predict(myglm1.2, type = "response", newdata = 
                             data.frame("diameter" = testingset[i,2]))
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
