# vertikale und horizontale Schnitte

ahorn <- diameter(dataahorn)
roteiche <- diameter(dataroteiche)
eiche <- diameter(dataeiche)
linde <- diameter(datalinde)
hainbuche <- diameter(datahainbuche)
esche <- diameter(dataesche)
buche <- diameter(databuche)

douglasie <- diameter(datadouglasie)
fichte <- diameter(datafichte)
kiefer <- diameter(datakiefer)


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




###### alle nadel vs alle laub


X <- rbind(ahorn, roteiche, eiche, linde, hainbuche, esche, buche, kiefer, fichte, douglasie)

cor(X[,2], X[,3])
myglm1.1<-glm(dummy ~ vertical, data = X, family = binomial(link
                                                             ="logit"))
myglm1.2<-glm(dummy ~ vertical, data = X, family = binomial(link
                                                            ="probit"))
myglm1.3<-glm(dummy ~ vertical, data = X, family = binomial(link
                                                            ="cloglog"))

pr1.1 <- predict.glm(myglm1.1, type = "response", newdata = 
                     data.frame("vertical" =  5))


myglm2<-glm(dummy ~ horizontal, data = X, family = binomial(link
                                                          ="logit"))

myglm3<-glm(dummy ~ horizontal + vertical, data = X, family = binomial(link
                                                          ="logit"))





##### nadelbäume und hainbuche unterscheiden 


X <- rbind(hainbuche, kiefer, douglasie, fichte)

myglm4<-glm(dummy ~ horizontal, data = X, family = binomial(link
                                                           ="logit"))

pr4 <- predict.glm(myglm4, type = "response", newdata = 
                      data.frame("horizontal" = 11))


####### buche gegen fichte


X <- rbind(buche, fichte)

myglm5<-glm(dummy ~ horizontal, data = X, family = binomial(link
                                                            ="logit"))
summary(myglm5)

(pr5 <- predict.glm(myglm5, type = "response", newdata = 
                     data.frame("horizontal" = 3)))



