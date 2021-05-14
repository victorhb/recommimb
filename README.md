# Recommendation of Techniques for Imbalanced Datasets

This is a recommender system of techniques for imbalanced datasets. It recommeds pre-processing and algorithmic-level techniques. The recommendation is based on a meta-learning approach using traditional meta-features and meta-features designed specifically for imbalanced datasets.

# Installation

This package is not available on CRAN but it can be installed with devtools.

```r
if (!require("devtools")) {
    install.packages("devtools")
}
devtools::install_github("victorhb/recommimb")
```

## Example of use

The command **recommimb** runs the meta-learning models and returns the top recommendations according to the data characteristics.

```r
data(arsenic_female_bladder)
recommimb(class ~ ., arsenic_female_bladder)
```
