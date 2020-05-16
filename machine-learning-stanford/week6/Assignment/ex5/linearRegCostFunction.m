function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;
Z = H-y;

grad = (1/m)* (X' * Z);
% Add regularization parameter to gradient
for j=2:n
    grad(j) = grad(j) + (lambda/m) * theta(j);
end

Z = Z .^ 2;
Z = sum(Z);
J = Z + (lambda * ( theta([2:end],:)' * theta([2:end],:) ) );
J = (1/(2*m)) * J;


% =========================================================================

grad = grad(:);

end
