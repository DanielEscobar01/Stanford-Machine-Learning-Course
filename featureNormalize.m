function [X_norm, mu, sigma] = featureNormalize(X) % FEATURENORMALIZE Normalizes the features in X 
	% 	Returns a normalized version of X where the mean value of each feature is 0 and the standard deviationis 1. 
	%	This is often a good preprocessing step to do when working with learning algorithms.

	% size(X, 2) returns the number of features or columns in X matrix
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));
   
	% Compute the mean of each column in X and put the value in mu vector
	mu= mean(X);
	% Compute the std of each column in X and put the value in sigma vector
	sigma=std(X);

	% Use i for columns and j for rows
	for i=1:size(X,2),
		for j=1:size(X,1),
		X_norm(j,i)=(X(j,i)-mu(1,i))/sigma(1,i); %This will set up the normalized values in the new matrix
		end;
	end;		 
 
end
