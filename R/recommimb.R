#' Recommendation of Techniques for Imbalanced Datasets
#'
#' This is a recommender system of techniques for imbalanced datasets. It recommeds
#' pre-processing and algorithmic-level techniques. The recommendation is based on a
#' meta-learning approach using traditional meta-features and meta-features designed
#' specifically for imbalanced datasets. 
#'
#' @family recommmender system
#' @param x A data.frame contained only the input attributes.
#' @param y A factor response vector with one label for each row/component of x.
#' @param top.list Number of recommended techniques. Default is 7 due to experimental evaluations.
#' @param formula A formula to define the class column.
#' @param data A data.frame dataset contained the input attributes and class.
#'  The details section describes the valid values for this group.
#' @param ... Further arguments passed to the summarization functions.
#' @details
#'  
#' @return A list of recommendations to be tested by the user
#'
#' @references
#'
#' @examples
#' ## Recommend techniques using formula
#' data(arsenic_female_bladder)
#' recommimb(class ~ ., arsenic_female_bladder)
#' 
#' @export
recommimb <- function(...) {
  UseMethod("recommimb")
}

#' @rdname recommimb
#' @export
recommimb.default <- function(x, y, top.list = 7, ...) {
  
  #models = readRDS("RF_models1.rds")
  #aux = readRDS("RF_models2.rds")
  print(list.files(system.file('extdata', package = 'recommimb'), full.names = TRUE))
  models = readRDS(list.files(system.file('extdata', package = 'recommimb'), full.names = TRUE)[1])
  aux = readRDS(list.files(system.file('extdata', package = 'recommimb'), full.names = TRUE)[2])
  models = c(models, aux)
  aux = NULL

  if(!is.data.frame(x)) {
    stop("data argument must be a data.frame")
  }

  if(is.data.frame(y)) {
    y <- y[, 1]
  }
  y <- as.factor(y)

  if(nrow(x) != length(y)) {
    stop("x and y must have same number of rows")
  }

  if(length(unique(y)) != 2){
    stop("It must be a binary classification task. Please check if y has two unique values.")
  }

  colnames(x) <- make.names(colnames(x))

  loadNamespace("randomForest")

  #Remove constant attributes
  #Factor
  x = x[,!sapply(1:ncol(x), FUN = function(coli){ length(levels(x[,coli])) == 1})]
  #Numeric
  x = x[,sapply(1:ncol(x), function(coli){ if(!is.numeric(x[,coli])) return(TRUE); return(var(x[,coli]) != 0) })]
  #Remove factor columns with more than 53 categories - because of random forest error
  x = x[,!sapply(1:ncol(x), FUN = function(coli){ length(levels(x[,coli])) > 53})]

  #Remove idenfier like attribute
  x = x[,sapply(1:ncol(x), FUN = function(i){ (length(unique(x[,i])) < nrow(x)) || ((is.numeric(x[,i]) && (sum(abs(x[,i] - floor(x[,i]))) > 0))) })]

  #Binarize
  x = binarize(x)

  #Renaming classes
  minority = names(which.min(table(y)))
  majority = names(which.max(table(y)))
  minority_i = which(levels(y) == minority)
  majority_i = which(levels(y) == majority)
  levels(y)[c(minority_i,majority_i)] <- c("1","0")
 
  #Meta-features calculation
  carac = characteristics(y)
  names(carac) = paste("carac.",names(carac),sep="")
  
  mfts.mfe = mfe::metafeatures(x,y, groups = c("general", "statistical", "infotheo","model","clustering"), summary = c("max","min","mean","median","sd","kurtosis","skewness"))
  mfts.mfe.aux = unlist(mfe::landmarking(x,y, folds = 5, summary = c("max","min","mean","median","sd","kurtosis","skewness")))
  names(mfts.mfe.aux) = paste("landmarking.", names(mfts.mfe.aux), sep="")
  
  mfts.mfe = c(mfts.mfe, mfts.mfe.aux)
  names(mfts.mfe) = paste("mfe.",names(mfts.mfe),sep="")
  
  mfts.ecol = ECoL::complexity(x,y)
  names(mfts.ecol) = paste("ecol.",names(mfts.ecol),sep="")
  
  mfts.imbcol = ImbCoL::complexity(x,y)
  names(mfts.imbcol) = paste("imbcol.",names(mfts.imbcol),sep="")
  
  ind = seq(1,length(mfts.imbcol)-1,2)
  mfts.imbcolgmean = sapply(ind, FUN= function(i){
    1-sqrt((1-mfts.imbcol[i])*(1-mfts.imbcol[i+1]))
  })
  aux = gsub("_partial\\.[0,1]","",names(mfts.imbcol)[ind])
  aux = gsub("imbcol","imbcolgmean",aux)
  names(mfts.imbcolgmean) = aux
  
  mtdata = c(carac, mfts.mfe, mfts.ecol, mfts.imbcol, mfts.imbcolgmean)

  mtdata = as.data.frame(t(mtdata))

  ps = unlist(sapply(names(models), function(f) {
    as.numeric(stats::predict(models[[f]], mtdata))
  }, simplify=FALSE))

  ps = as.data.frame(ps)
  ps = cbind(ps, set_recommendations())
  colnames(ps) = c("gmean.prediction", "algorithms")

  ps = ps[order(-ps[,1]),]
  rownames(ps) = NULL

  ret = recommimb.recommendations(recommendations = "", performance.predictions = ps)
  ret$set.top.list(top.list)

  ret
}

#' @rdname recommimb
#' @export
recommimb.formula <- function(formula, data, top.list = 7, ...) {
  if(!inherits(formula, "formula")) {
    stop("method is only for formula datas")
  }

  if(!is.data.frame(data)) {
    stop("data argument must be a data.frame")
  }

  modFrame <- stats::model.frame(formula, data)
  attr(modFrame, "terms") <- NULL

  recommimb.default(modFrame[, -1], modFrame[, 1], top.list, ...)
}

recommimb.recommendations <- setRefClass("recommimb.recommendations", 
  fields = list(recommendations = "character", 
                performance.predictions = "data.frame"), 
  methods = list(show = function(){
                  cat(recommendations)
            },
            set.top.list = function(top.list){
                  recommendations <<- paste(c("Based on the data characteristics of your dataset, you should consider testing the following:",
                  as.character(gsub(" \\| ","\n",performance.predictions[1:top.list,2])),"For detailed description of the predicted performances, please check $performance.predictions\n"),collapse="\n\n")
            })
  )

characteristics <- function(y)
{
  tb <- table(y)
  names(tb) <- paste("size",names(tb),sep=".")
  return(c(tb, "perc" = tb/sum(tb)))
}

form <- function(x) {
  att <- paste(colnames(x), collapse="+")
  stats::formula(paste("~ 0 +", att, sep=" "))
}

binarize <- function(x) {
  data.frame(stats::model.matrix(form(x), x))
}

imputation <- function(data) {
  rbind(replace(data, !is.finite(data) , 0))
}

set_recommendations <- function(){

  a = "Preprocessing technique:  None | Classification algorithm: SVM (radial)"
  a = c(a, "Preprocessing technique:  None | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  None | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: JRip")

  a = c(a, "Preprocessing technique:  None | Classification algorithm: one-class SVM (radial)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: one-class SVM (linear)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: one-class SVM (polynomial)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: one-class SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  None | Classification algorithm: Isolation Forest")

  a = c(a, "Preprocessing technique:  None | Classification algorithm: Adaboost (C5.0)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Adaboost (SVM)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random Underbagging (C5.0)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random Underbagging (SVM)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random SMOTEbagging (C5.0)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random SMOTEbagging (SVM)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random Underboosting (C5.0)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random Underboosting (SVM)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random SMOTEboosting (C5.0)")
  a = c(a, "Preprocessing technique:  None | Classification algorithm: Random SMOTEboosting (SVM)")


  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: SVM (radial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: JRip")

  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: one-class SVM (radial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: one-class SVM (linear)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: one-class SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: one-class SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Random Undersampling | Classification algorithm: Isolation Forest")

  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: SVM (radial)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  Random Oversampling | Classification algorithm: JRip")

  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: SVM (radial)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  SMOTE | Classification algorithm: JRip")

  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: SVM (radial)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  Borderline-SMOTE | Classification algorithm: JRip")


  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: SVM (radial)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: SVM (linear)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: SVM (polynomial)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: class weighted SVM (radial)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: class weighted SVM (linear)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: class weighted SVM (polynomial)")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: class weighted SVM (sigmoid)")

  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: Random Forest")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: class weighted Random Forest")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: KNN")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: Naive Bayes")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: J48")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: MLP")
  a = c(a, "Preprocessing technique:  ADASYN | Classification algorithm: JRip")

  return(a)
}
