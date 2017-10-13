function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum1 = 0;
    sum2 = 0;
    for i = 1:m
        a = (X(i,:) * theta - y(i)) * X(i,1);
        b = (X(i,:) * theta - y(i)) * X(i,2);
        sum1 = sum1 + a;
        sum2 = sum2 + b;
    end
    
    theta(1) = theta(1) - (alpha / m) * sum1;
    theta(2) = theta(2) - (alpha / m) * sum2;
    
    
%   for j = 1:size(X,2)   
%      sum = 0;
%      for i = 1:m
%        a = (X(i,:) * theta - y(i)) * X(i,j);
%        sum = sum + a;
%      end
%      theta(j) = theta(j) - (alpha / m) * sum;
%    end
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
