data=readtable("data.csv");
X = data{:, { 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'}};
Y = data.price;

fprintf("The maximum value of a house is $%2f",max(Y))

rng(30)

splitIndex = round(0.70 * size(data));

X_train = X(1:splitIndex, :);
Y_train = Y(1:splitIndex, :);

X_test = X(splitIndex+1:end, :);
Y_test = Y(splitIndex+1:end, :);
model=fitlm(X_train,Y_train);

pred=predict(model,X_test);

rmse = sqrt(mean((Y_test - pred).^2));  


r_squared = 1 - (sum((Y_test - pred).^2) / sum((Y_test - mean(Y_test)).^2));  
fprintf("The error in the predicted model is  %.2f ",r_squared*100)
fprintf("The error in the predicted model for a given house is approximately given different factors affecting at that time  $%.2f ",rmse)

% Scatter plot of actual vs. predicted values
scatter(Y_test, pred);
xlabel('Actual Prices');
ylabel('Predicted Prices');
title('Actual vs. Predicted Prices');
grid on;

% Residual plot
residuals = Y_test - pred;
figure;
scatter(pred, residuals);
xlabel('Predicted Prices');
ylabel('Residuals');
title('Residual Plot');
grid on;

% Histogram of residuals
figure;
histogram(residuals, 20);  % Adjust the number of bins as needed
xlabel('Residuals');
ylabel('Frequency');
title('Histogram of Residuals');
grid on;
axis auto;

