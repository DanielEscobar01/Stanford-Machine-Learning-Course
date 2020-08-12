function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters) %This function Performs gradient descent to learn theta

    %theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by  taking num_iters gradient steps with learning rate alpha
    m = length(y); % Number of training examples
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters

        % X dimension is (97,2)
        % Theta dimension is (2,1)
        % Y dimension is (97,1)
        % ThetaUpdated equals theta - (alpha divided by m) times 
        % In the equation Htheta-y is equal to X*theta-y
        % (X*theta) gets a vector with the values of predictions
        % Then -y results in a vector with the difference between real value and predictions
        % (X*theta-y) gives us a vector with dimension (97,1)
        % In order to satisface the equation we have to put X transpose first to get theta with dimension (2,1)

        theta = theta - alpha/m*(X'*(X*theta-y));

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end
end
