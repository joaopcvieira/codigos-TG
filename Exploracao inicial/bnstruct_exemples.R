# install.packages("bnstruct")
library(bnstruct)


#Package para construir redes Bayesianas e imputação de dados faltantes

# Example ASIA

dataset <- asia()

print(dataset)
show(dataset) #same as print
net<-BN(dataset)
net

#aprendendo a estrutura e parâmetros

net<- learn.network(dataset)
dag(net)
cpts(net) #Conditional Probabilities CPT
plot(net) # regular DAG

plot(net, method="qgraph",
     label.scale.equal=T,
     node.width = 1.6,
     plot.wpdag=F)

engine <- InferenceEngine(net)
engine

###################
dataset <- asia()
net <- learn.network(dataset)
# same as instantiations
interventions <- list(intervention.vars=c(3),
                      intervention.vals=c(1))
engine <- InferenceEngine(net, interventions = interventions)
marginals(engine)
cpts(net)
test.updated.bn(engine) # TRUE
get.most.probable.values(updated.bn(engine))

interventions

###########
dataset <- child()
# learning with available cases analysis, MMHC, BDeu
net <- learn.network(dataset)
plot(net)
# learning with imputed data, MMHC, BDeu
imp.dataset <- impute(dataset)
net <- learn.network(imp.dataset, use.imputed.data = TRUE)
plot(net)

net <- learn.network(dataset, algo = "sem",
                      scoring.func = "BDeu",
                      initial.network = net,
                      struct.threshold = 10,
                      param.threshold = 0.001)
plot(net)


# we update the probabilities with EM from the raw dataset,
# starting from the first network
engine <- InferenceEngine(net)


results <- em(engine, dataset)
updated.engine <- results$InferenceEngine; updated.engine
updated.dataset <- results$BNDataset; updated.dataset


###########
dataset <- asia()
dataset <- bootstrap(dataset)
net <- learn.network(dataset, bootstrap = TRUE)
plot(net)
