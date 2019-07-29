# --------------------------------------------------------
# title: "ISPOR Webinar : Machine Learning for Prediction"
# author: "Dave Vanness"
# date: "July 2019"
# --------------------------------------------------------

# --------------------------------------------------------
# Load libraries, clear memory and define helper functions

library(glmnet)               #Elastic net regression (LASSO, Ridge and in-between)
library(RColorBrewer)         #Finer color control for figures
library(rpart)                #Recursive partitioning (trees)
library(rpart.plot)           #Package for plotting recursive partitions
library(caret)                #Helper package for machine learning
library(Rborist)              #Random Forest package with additional functions
library(doParallel)           #Allows use of multiple cores on a single machine
library(xgboost)              #Extreme gradient boosting package
library(pdp)                  #Package for making partial dependence plots - including 2-way
rm(list = ls())
par.orig = par()
logit = function(x) {
  1 / (1 + exp(-x))
}

# --------------------------------------------------------
# Create Simulated Dataset

N_pop = 100000                #Number in population
sampled = 1:2000              #First 2000 of true population are sampled
ierr = 1                      #Irreducible variation 0:0%, 1:50%, 2:66%, 3:75%
set.seed(seed = 20190723)     #Set seed for reproducibility

#True Effects
X_n = matrix(data = rnorm(n = N_pop * 8),
             nrow = N_pop,
             ncol = 8)                        #Normally distributed predictors
X_g = matrix(
  data = rgamma(
    n = N_pop * 7,
    shape = rep(c(5, 10, 20), 2),
    scale = rep(1 / c(5, 10, 20), 2)
  ),
  nrow = N_pop,
  ncol = 7
)                                             #Gamma distributed predictors
X_b = matrix(rbinom(N_pop * 3, 1, .5), 
             nrow = N_pop, ncol = 3)          #Binomial predictors
X_c = cbind(                                  #Categorical predictors
  sample(
    LETTERS[1:4],
    N_pop * 1,
    replace = TRUE,
    prob = c(0.25, 0.25, 0.25, 0.25)
  ),
  sample(
    LETTERS[21:23],
    N_pop * 1,
    replace = TRUE,
    prob = c(1 / 3, 1 / 3, 1 / 3)
  )
)                                             
#False Targets (equal number of each type)
X_n_NULL = matrix(data = rnorm(n = N_pop * 8),
                  nrow = N_pop,
                  ncol = 8)                   #Normally distributed predictors
X_g_NULL = matrix(
  data = rgamma(
    n = N_pop * 7,
    shape = rep(c(5, 10, 20), 2),
    scale = rep(1 / c(5, 10, 20), 2)
  ),
  nrow = N_pop,
  ncol = 7
)                                             #Gamma distributed predictors
X_b_NULL = matrix(rbinom(N_pop * 3, 1, .5), 
                  nrow = N_pop, ncol = 3)     #Binomial predictors
X_c_NULL = cbind(
  sample(
    LETTERS[1:4],
    N_pop * 1,
    replace = TRUE,
    prob = c(0.25, 0.25, 0.25, 0.25)
  ),
  sample(
    LETTERS[21:23],
    N_pop * 1,
    replace = TRUE,
    prob = c(1 / 3, 1 / 3, 1 / 3)
  )
)                                                                      

f = matrix(data = NA, nrow = N_pop, ncol = 20) #Generate functionals from predictors

f[, 1] = X_n[, 1]                                                 #Linear terms
f[, 2] = X_n[, 2]
f[, 3] = X_n[, 3]
f[, 4] = 3 * X_g[, 1]
f[, 5] = 3 * X_g[, 2]
f[, 6] = 2 * X_b[, 1]
f[, 7] = 2 * X_b[, 2]
f[, 8] = 0.75 * (-1 * (X_c[, 1] == "A") +-2 * (X_c[, 1] == "B") 
                 + 2 * (X_c[, 1] == "C") + 1 * (X_c[, 1] == "D"))
f[, 9] = 0.5 * (1 * (X_c[, 2] == "U") + 2 * (X_c[, 2] == "V") 
                - 2 * (X_c[, 2] == "W"))
f[, 10] = 0.75 * (1 * X_n[, 4] +-.5 * X_n[, 4] ^ 2)               #Quadratic terms
f[, 11] = 0.5 * X_n[, 5] + .5 * X_n[, 5] ^ 2
f[, 12] = 2*(2.5*X_g[,3]^2 - X_g[,3]^3)                           #Non-linear functions
f[, 13] = 2*cos(3 * X_n[, 6])
f[, 14] = 1.75 * abs(X_n[, 7])
f[, 15] = 1 * X_n[, 8] * X_b[, 3] - 1 * X_n[, 8]*(1-X_b[, 3])     #Interactions
f[, 16] = X_g[,4] -1.5 * X_g[, 4] * X_g[, 5]*(X_g[,5]>1)
f[, 17] = X_n[, 5] * X_g[, 2]                                     #Heterogeneity
f[, 18] = 1 * X_n[, 2] * X_b[, 2] - 1 * X_n[, 2] * (1-X_b[, 2])
f[, 19] = -1.5 * X_n[, 3] * X_g[, 6] * (X_g[,6 ] < 0.8) 
                + 1.5*(X_n[, 3] * X_g[, 6] * (X_g[,6 ] > 1.2))
f[, 20] = 0.75 * ((X_c[, 2] == "U") * X_g[, 7] 
                  -1 * (X_c[, 2] == "V") * X_g[, 7] 
                  + 2 * (X_c[, 2] == "W") * X_g[, 7])
selector = c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)       #Use selector to turn functionals 
                                                            #on and off to test performance
Y.raw = f %*% selector                                      #Combine into the raw outcome variable

Y.c = Y.raw + rnorm(n = N_pop,
                    mean = 0,
                    sd = ierr * sd(Y.raw))                  #Add irreducible variation (unpredictable noise)
all.c = data.frame(Y.c, X_n, X_g, X_b, X_c, X_n_NULL, X_g_NULL, X_b_NULL, X_c_NULL) #Data frame of 100,000 population
names(all.c) = c("Y.c","N1_L","N2_LxB2", "N3_LxG6", "N4_Q", "N5_QxG2", "N6_COS", "N7_ABS", "N8_LxB3",
                 "G1_L", "G2_LxN5", "G3_CUBIC", "G4_LxG5", "G5_LxG4", "G6_LxN3", "G7_LxC2",
                 "B1_L", "B2_LxN2", "B3_LxN8", 
                 "C1_L", "C2_LxG7",
                 "N1_NULL","N2_NULL","N3_NULL","N4_NULL","N5_NULL","N6_NULL","N7_NULL","N8_NULL",
                 "G1_NULL","G2_NULL","G3_NULL","G4_NULL","G5_NULL","G6_NULL","G7_NULL",
                 "B1_NULL","B2_NULL","B3_NULL",
                 "C1_NULL", "C2_NULL")

#Convert binary and categorical predictors into factors
all.c[, "B1_L"] = as.factor(all.c[, "B1_L"])
all.c[, "B2_LxN2"] = as.factor(all.c[, "B2_LxN2"])
all.c[, "B3_LxN8"] = as.factor(all.c[, "B3_LxN8"])
all.c[, "B1_NULL"] = as.factor(all.c[, "B1_NULL"])
all.c[, "B2_NULL"] = as.factor(all.c[, "B2_NULL"])
all.c[, "B3_NULL"] = as.factor(all.c[, "B3_NULL"])

all.s = sparse.model.matrix(Y.c ~ ., data = all.c)[, -1]     #Population sparse matrix - converts factors to dummies
all.d = cbind(Y.c,as.data.frame(as.matrix(all.s)))           #Population Data frame with dummies converted
sampled_data.c = all.c[sampled, ]                            #Sample for training : Categorical
sampled_data.s = all.s[sampled, ]                            #Sample for training : Sparse
sampled_data.d = all.d[sampled, ]                            #Sample for training : Dummies
val_data.c = all.c[-sampled, ]                               #Validation set (pop minus sample) : Categorical
val_data.d = all.d[-sampled, ]                               #Validation set : Dummies
Xvarnames.c = names(all.c[, -1])                             #List of predictor names
Xvarnames.d = names(all.d[, -1])                             #List of predictor names (differs because of dummies)
perfect.RMSE = sqrt(mean((Y.c - Y.raw) ^ 2))                 #RMSE if predicted Y.raw perfectly: irreducible error
random.RMSE = sqrt(mean((Y.c - Y.c[sample(x = 1:N_pop,
                                          size = N_pop,
                                          replace = TRUE)]) ^ 2)) #RMSE if random guess from sample
mean.RMSE = sqrt(mean((Y.c - mean(Y.c[sampled])) ^ 2))       #RMSE we would get by using the sample population mean

# --------------------------------------------------------
# Linear Regression
lm.Y.c = lm(Y.c ~ ., data = sampled_data.c)                 #lm() linear model
Y.c.val.lm = predict(object = lm.Y.c, newdata = val_data.c) #Use linear model to predict in validation dataset
Y.c.lm.err = val_data.c[, "Y.c"] - Y.c.val.lm               #Calculate prediction error
Y.c.lm.RMSE = sqrt(mean((Y.c.lm.err ^ 2)))                  #Calculate RMSE

# Generate figures
pdf("output/fitsofar0.pdf", width = 7, height = 8)
barplot(
  c(random.RMSE, mean.RMSE, Y.c.lm.RMSE),
  col = "steelblue",
  names.arg = c("Random", "Sample Mean", "Linear Reg"),
  xlab = "Square Root of Mean Squared Error (RMSE)",
  cex.names = 0.8
)
abline(h = perfect.RMSE)
text(x = .7,
     y = perfect.RMSE + .2,
     "Lower Bound",
     cex = 0.8)
dev.off()

# --------------------------------------------------------
# LASSO Regression using glmnet

lasso.Y.c = glmnet(x = as.matrix(sampled_data.d[,-1]), y = as.matrix(sampled_data.d[,"Y.c"]), alpha = 1)

#Perform cross-validation to set regularization parameter to minimize bias
set.seed(seed = 20190723) #Set seed for reproducible cross-validation
cv.lasso.Y.c = cv.glmnet(x = as.matrix(sampled_data.d[,-1]), y = as.matrix(sampled_data.d[,"Y.c"]), alpha=1, nfolds = 5)

#Assess performance
Y.c.val.lasso = predict.cv.glmnet(object = cv.lasso.Y.c,newx = as.matrix(val_data.d[,-1]), s="lambda.1se") 
Y.c.lasso.err = val_data.c[,"Y.c"] - Y.c.val.lasso 
Y.c.lasso.RMSE = sqrt(mean(Y.c.lasso.err^2))

#Note: this color palette is set so that the false target predictors are all in red
mypalette = c(brewer.pal(8,"Dark2"),brewer.pal(8,"Accent")[-6],brewer.pal(8,"Pastel1"),rep("red",23))

#Generate Figures
pdf("output/lasso.pdf", width = 9, height = 8)
par(mar = par.orig$mar + c(0, 0, 0, 6), xpd = TRUE)
plot(
  lasso.Y.c,
  xvar = "lambda",
  label = FALSE,
  lwd = 2,
  col = mypalette
)
abline(v = log(cv.lasso.Y.c$lambda.1se))
abline(v = log(cv.lasso.Y.c$lambda.min), lty = "dotted")
legend(
  "topright",
  inset = c(-.175, 0),
  legend = c(Xvarnames.d),
  fill = mypalette,
  cex = .695,
  xpd = TRUE
)
dev.off()

pdf("output/cv_lasso.pdf", width = 10, height = 8)
plot(cv.lasso.Y.c)
abline(v = log(cv.lasso.Y.c$lambda.1se))
dev.off()

pdf("output/fitsofar1.pdf", width = 7, height = 8)
barplot(
  c(random.RMSE, mean.RMSE, Y.c.lm.RMSE, Y.c.lasso.RMSE),
  col = "steelblue",
  names.arg = c("Random", "Sample Mean", "Linear Reg", "LASSO"),
  xlab = "Square Root of Mean Squared Error (RMSE)",
  cex.names = 0.8
)
abline(h = perfect.RMSE)
text(x = .7,
     y = perfect.RMSE + .2,
     "Lower Bound",
     cex = 0.8)
abline(h = Y.c.lm.RMSE, lty = "dotted")
text(x = 3.1,
     y = Y.c.lm.RMSE + .2,
     "Linear RMSE",
     cex = .7)
dev.off()

# --------------------------------------------------------
# Recursive Partitioning (Regression Trees)

set.seed(seed = 20190723)     #Set seed for reproducible cross-validation
cv.rpart.Y.C = rpart(Y.c~.,data = sampled_data.c,control = rpart.control(cp=.005, xval = 10)) #10-fold Cross-Validation

#Prune the tree using the complexity parameter selected from cross-validation
cptable = printcp(cv.rpart.Y.C)
best.cp = cptable[which.min((min(cptable[,4])+cptable[,5])<cptable[,4]),1]+1E-7           #Implementing the 1SE Rule
best.cp <- cptable[which.min(cptable[,4]),1]+1E-7                                         #Min-Xerror Rule
rpart.Y.c = prune.rpart(tree = cv.rpart.Y.C,cp = best.cp)                                 #Pruned tree

#Assess performance
Y.c.val.rpart = predict(object = rpart.Y.c, newdata = val_data.c)
Y.c.rpart.err = val_data.c[, "Y.c"] - Y.c.val.rpart
Y.c.rpart.RMSE = sqrt(mean(Y.c.rpart.err ^ 2))

pdf("output/plotcp.pdf", width = 10, height = 8)
plotcp(cv.rpart.Y.C, upper = "size")
dev.off()

pdf("output/big_tree.pdf", width = 10, height = 8)
rpart.plot(cv.rpart.Y.C)
dev.off()

pdf("output/pruned_tree.pdf",
    width = 10,
    height = 8)
rpart.plot(rpart.Y.c)
dev.off()

pdf("output/fitsofar2.pdf",width = 7,height = 8)
barplot(c(random.RMSE,mean.RMSE,Y.c.lm.RMSE,Y.c.lasso.RMSE,Y.c.rpart.RMSE), col="steelblue", names.arg = c("Random","Sample Mean", "Linear Reg", "LASSO", "Rpart"),xlab="Square Root of Mean Squared Error (RMSE)",cex.names = 0.7)
abline(h=perfect.RMSE)
text(x=.7,y=perfect.RMSE+.2,"Lower Bound",cex = .7)
dev.off()

# --------------------------------------------------------
# Random Forest

#Uses the caret package to conduct algorithm optimization
nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel

rf.trainControl = trainControl(
  ## 5-fold CV
  method = "repeatedcv",
  # Five-fold cross-validation
  number = 5,
  # Repeated five times
  repeats = 5,
  allowParallel = TRUE
)

#Parameters to tune using caret grid search
rf.grid = expand.grid(
  # Number of randomly selected parameters used to grow each tree    
  predFixed = seq(10,40,5),
  # Controls tree size by putting floor on N for nodes to be split       
  minNode = c(1, 5, 10, 20)
)     

set.seed(20190723)
rf.train = train(
  Y.c ~ .,
  # Model considers all variables
  data = sampled_data.c,
  method = "Rborist",
  trControl = rf.trainControl,
  tuneGrid = rf.grid,
  nThread = 1,
  # Fit 1,000 trees
  nTree = 1000                          
)

#Assess performance
Y.c.val.rf = predict(object = rf.train,newdata = val_data.c)
Y.c.rf.err = val_data.c[,"Y.c"] - Y.c.val.rf
Y.c.rf.RMSE = sqrt(mean(Y.c.rf.err^2))

#Generate figures
pdf("output/rf.train.pdf", width = 10, height = 8)
plot(rf.train)
dev.off()

pdf("output/rf.importance.pdf",
    width = 8,
    height = 14)
plot(varImp(rf.train), cex = .5)
dev.off()

pdf("output/fitsofar3.pdf", width = 10, height = 8)
barplot(
  c(
    random.RMSE,
    mean.RMSE,
    Y.c.lm.RMSE,
    Y.c.lasso.RMSE,
    Y.c.rpart.RMSE,
    Y.c.rf.RMSE
  ),
  col = "steelblue",
  names.arg = c(
    "Random",
    "Sample Mean",
    "Linear Reg",
    "LASSO",
    "Rpart",
    "Random Forest"
  ),
  xlab = "Square Root of Mean Squared Error (RMSE)",
  cex.names = 0.7
)
abline(h = perfect.RMSE)
text(x = .7,
     y = perfect.RMSE + .2,
     "Lower Bound",
     cex = .7)
dev.off()

rf.pdp = list()
rf.pdp[[1]] =
  partial(
    object = rf.train,
    pred.var = c("N8_LxB3", "B3_LxN8"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
rf.pdp[[2]] =
  partial(
    object = rf.train,
    pred.var = c("G4_LxG5", "G5_LxG4"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
rf.pdp[[3]] =
  partial(
    object = rf.train,
    pred.var = c("N5_QxG2", "G2_LxN5"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
rf.pdp[[4]] =
  partial(
    object = rf.train,
    pred.var = c("N3_LxG6", "G6_LxN3"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
rf.pdp[[5]] =
  partial(
    object = rf.train,
    pred.var = c("N2_LxB2", "B2_LxN2"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
rf.pdp[[6]] =
  partial(
    object = rf.train,
    pred.var = c("C2_LxG7", "G7_LxC2"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )

pdf(file = "output/rf_twoway.pdf")
par(mfrow = c(2,3))
lapply(rf.pdp,FUN = plotPartial)
dev.off()

stopCluster(cl)


# --------------------------------------------------------
# Extreme Gradient Boosting

nc = parallel::detectCores()  #Detects how many cores current machine has
cl = makePSOCKcluster(nc-1)   #Set number of cores equal to machine number minus one
registerDoParallel(cl)        #Set up parallel

xgb.trainControl = trainControl(
  ## 5-fold CV
  method = "repeatedcv",
  number = 5,
  # Five-fold cross-validation
  repeats = 5,
  # Repeated five times
  allowParallel = TRUE
)

# Tuning will occur in stages due to number of tuneable parameters

# Round 1: find learning rate eta that works for a reasonable number of trees (500) and tree depth
xgb.grid.1 = expand.grid(
  nrounds = c(1,10,50,100,200,300,400,500),
  max_depth = c(1,2,3,4),
  eta = c(.05,.1,.2),
  gamma = 0,
  min_child_weight = 1,
  colsample_bytree = 1,
  subsample = 1
)

set.seed(20190723)
xgb.train_1 = train(Y.c ~ ., data = sampled_data.c, 
                    method = "xgbTree", 
                    trControl = xgb.trainControl,
                    tuneGrid = xgb.grid.1,
                    verbose = TRUE,
                    nthread = 1)

pdf("output/xgb.train_1.pdf",width = 10, height = 8)
plot(xgb.train_1)
dev.off()

# Round 2: Using best parameters from round 1, search for tree size by constraining minimum N to split a node
xgb.grid.2 = expand.grid(
  nrounds = 500,
  max_depth = 2,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1,seq(10,150,10)),
  subsample = 1
)

set.seed(20170723)
xgb.train_2 = train(Y.c ~ ., data = sampled_data.c, 
                    method = "xgbTree", 
                    trControl = xgb.trainControl,
                    tuneGrid = xgb.grid.2,
                    verbose = TRUE,
                    nthread = 1)

pdf("output/xgb.train_2.pdf",width = 10, height = 8)
plot(xgb.train_2)
dev.off()

# Round 3: Using best parameters from round 2, consider bootstrap subsample and random feature selection
xgb.grid.3 = expand.grid(
  nrounds = 500,
  max_depth = 2,
  eta = .05,
  gamma = 0,
  colsample_bytree = seq(.1,1,.1),
  min_child_weight = 60,
  subsample = seq(.5,1,.1)
)

set.seed(20170723)
xgb.train_3 = train(Y.c ~ ., data = sampled_data.c, 
                    method = "xgbTree", 
                    trControl = xgb.trainControl,
                    tuneGrid = xgb.grid.3,
                    verbose = TRUE,
                    nthread = 1)

pdf("output/xgb.train_3.pdf",width = 10, height = 8)
plot(xgb.train_3)
dev.off()

# Round 4: Using best parameters from round 3, consider slower learning rates and increasing number of trees
xgb.grid.4 = expand.grid(
  nrounds = c(1,50,100,seq(200,4000,100),seq(5000,25000,5000)),
  max_depth = 1,
  eta = c(.01,.025,.05,.1,.2),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 60,
  subsample = .6
)

set.seed(20170723)
xgb.train_4 = train(Y.c ~ ., data = sampled_data.c, 
                    method = "xgbTree", 
                    trControl = xgb.trainControl,
                    tuneGrid = xgb.grid.4,
                    verbose = TRUE,
                    nthread = 1)

pdf("output/xgb.train_4.pdf",
    width = 10,
    height = 8)
plot(xgb.train_4)
dev.off()

# Assess performance

Y.c.val.boost = predict(object = xgb.train_4, newdata = val_data.c)
Y.c.boost.err = val_data.c[, "Y.c"] - Y.c.val.boost
Y.c.boost.RMSE = sqrt(mean(Y.c.boost.err ^ 2))

#Generate figures
pdf("output/xgb.importance.pdf",
    width = 8,
    height = 10)
plot(varImp(xgb.train_4))
dev.off()

pdf("output/fitsofar4.pdf", width = 10, height = 8)
barplot(
  c(
    random.RMSE,
    mean.RMSE,
    Y.c.lm.RMSE,
    Y.c.lasso.RMSE,
    Y.c.rpart.RMSE,
    Y.c.rf.RMSE,
    Y.c.boost.RMSE
  ),
  col = "steelblue",
  names.arg = c(
    "Random",
    "Sample Mean",
    "Linear Reg",
    "LASSO",
    "Rpart",
    "Random Forest",
    "XGBoost"
  ),
  xlab = "Square Root of Mean Squared Error (RMSE)",
  cex.names = 0.7
)
abline(h = perfect.RMSE)
text(x = .7,
     y = perfect.RMSE + .2,
     "Lower Bound",
     cex = .7)
abline(h = Y.c.boost.RMSE, lty = "dotted")
text(x = 7.9,
     y = Y.c.boost.RMSE + .2,
     "XGBoost RMSE",
     cex = .5)
dev.off()

pdf("output/xgb.SHAP.pdf",
    width = 12,
    height = 8)
xgb.plot.shap(
  data = sampled_data.s,
  model = xgb.train_4$finalModel,
  top_n = 12,
  n_col = 4
)
dev.off()
xgb.pdp = list()
xgb.pdp[[1]] =
  partial(
    object = xgb.train_4,
    pred.var = c("N8_LxB3", "B3_LxN8"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
xgb.pdp[[2]] =
  partial(
    object = xgb.train_4,
    pred.var = c("G4_LxG5", "G5_LxG4"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
xgb.pdp[[3]] =
  partial(
    object = xgb.train_4,
    pred.var = c("N5_QxG2", "G2_LxN5"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
xgb.pdp[[4]] =
  partial(
    object = xgb.train_4,
    pred.var = c("N3_LxG6", "G6_LxN3"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
xgb.pdp[[5]] =
  partial(
    object = xgb.train_4,
    pred.var = c("N2_LxB2", "B2_LxN2"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )
xgb.pdp[[6]] =
  partial(
    object = xgb.train_4,
    pred.var = c("C2_LxG7", "G7_LxC2"),
    plot = FALSE,
    chull = TRUE,
    plot.engine = "ggplot2"
  )

pdf(file = "output/xgb_twoway.pdf")
par(mfrow = c(2,3))
lapply(xgb.pdp,FUN = plotPartial)
dev.off()

stopCluster(cl)