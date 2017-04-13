function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% Note: grad should have the same dimensions as theta
grad = zeros(size(theta)); % same dimensions as theta

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.

hypothesis = sigmoid(X * theta);

% You should set J to the cost.

J = (-1/m) * sum(y .* log(hypothesis) +(1 - y).*log(1 - hypothesis));

% Compute the partial derivatives and set grad to the partial
%  -> derivatives of the cost w.r.t. each parameter in theta

%
%

for i = 1 : m
    % hypothesis = mx1 column vector
    % y = mx1 column vector
    % X = mxn matrix
    grad = grad + (hypothesis(i) - y(i) ) * X(i, :)'; % X(i, :) = is the mth row of the matrix A
end;

grad = (1 / m) * grad;



% =============================================================

end
