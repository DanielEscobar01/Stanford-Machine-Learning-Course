function J = computeCost(X, y, theta) %This function compute the cost for linear regression
	% Theta is a vector with [theta0, theta1] its order is (2x1)
	% X order is (97x2) so we have to transpose in order to multiply with theta
	% X and y are matrices whose rows represent the examples from the training set

	m = length(y); % M = Number of training examples

	% (X*theta) gets a vector with the values of predictions
	% Then -y results in a vector with the difference between real value and predictions
	% After that, we power by 2 every single difference stored in the vector
	% Then with sum(...) we get the value of the summarize of every single error 
	J = 1/(2*m) * sum(((X*theta)-y).^2);
end