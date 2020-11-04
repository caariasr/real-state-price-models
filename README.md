# real-state-price-models
This is a simple approach Automated Price Valuation (AVM) for sale and rent apartments and classification for whether the apartment will be transacted after a month
The AVM serves predictions through a simple docker-flask API.
The instructions to test the API are [here](/api/README.md).

## Results (High Level)

To properly visualize the notebook please use this link [notebook](https://nbviewer.jupyter.org/github/caariasr/real-state-price-models/blob/master/notebooks/dubai_avm.ipynb)

The objective of the Automated Valuation Model is to predict house prices
for apartments both for sale and rent. The data comes from the property finder
website. To predict the property price, I used different  attributes about the
 property including location, square foot, etc. 
 
 The final model for the AVM is a Neural Network. In order to validate the
 results, I removed 30% of the provided listings from the model
 completely. After I trained and tuned the model I
 predicted the logarithm of the price for the hidden listings. In the following
 chart you can see the comparison for each listing's predicted value (x-axis) with
 respect to its actual value (y-axis).
  
![Predicted vs Actual Plot](/api/static/png/pred_vs_act.png)

Other meaningful metrics of the model performance:

1. Root mean square error (RMSE) for all listings: 864,428 AED
2. RMSE listings on sale: 864,428 AED
3. RMSE listings for rent: 46,080 AED

For comparison of performance between sale and rent listings is better to use
the Mean Absolute Percentage Error (MAPE):

1. MAPE for all listings: 17.94 %
2. MAPE listings on sale: 16.86 %
3. MAPE listings for rent: 19.15 %
