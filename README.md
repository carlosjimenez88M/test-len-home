# Lending Home Test


In this git you cand find this products 

|Solution|Model|
|-------:|-----|
|How is LendingHome doing compared to other lenders?|[Visual Interpretation](https://github.com/carlosjimenez88M/test-len-home/blob/master/lender.md)|
|Predict transactions|[Model](https://github.com/carlosjimenez88M/test-len-home/blob/master/prediction_model.md)|

## Approaches

### How is LendingHome doing compared to other lenders?

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/lender_files/figure-gfm/unnamed-chunk-5-1.png)

### Anomalies Detection

I evaluate this dataset with two techniques: Structural Changes and grubbs test 

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-4-1.png)

Whit this I understand the outliers problem

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-7-1.png)

And with this I create soft filter for anoms data

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-8-1.png)

## Statistical application to Amount Transactions

In this case I use probabilistic model (log normal) to drop some dates (years)

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-15-1.png)

And for this I worked with Year>1930.

And finally I worked with missing info

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-12-1.png)

## Machine Learning Model

I Worked with two models : Xgboost (for weaks correlations) and decision tree (experimental), with this outputs 

### XGboost


fine tuning process 

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-18-1.png)


![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-21-1.png)

### Interpretability shape

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-23-1.png)


And the most important features

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-22-1.png)

### decision tree 


Perforamnce model

![](https://github.com/carlosjimenez88M/test-len-home/raw/master/prediction_model_files/figure-gfm/unnamed-chunk-26-1.png)