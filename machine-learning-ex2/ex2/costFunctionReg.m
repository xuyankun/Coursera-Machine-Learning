function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum_j = 0;

for i = 1:m
  iter_j = y(i) * log(sigmoid(X(i,:)*theta)) + (1 - y(i)) * ...
  log(1-sigmoid(X(i,:)*theta));
  sum_j = sum_j + iter_j;
  
end

J =  lambda * sum(theta(2:end).^2)/(2*m)  - sum_j / m ;

sum_g = 0;

for i = 1:m
    iter_g = (sigmoid(X(i,:)*theta) - y(i)) * X(i,1)
    sum_g = sum_g + iter_g;
end
grad(1) = sum_g / m;

for j = 2:size(theta)
  sum_g = 0;
  for i = 1:m
    iter_g = (sigmoid(X(i,:)*theta) - y(i)) * X(i,j)
    sum_g = sum_g + iter_g;
  end
  grad(j) = (sum_g + lambda * theta(j)) / m ;

end




% =============================================================

end
