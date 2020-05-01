function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % 5000
num_labels = size(Theta2, 1); %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add x0=1 for layer 1
X = [ ones(m,1) X]; % 5000*401

% Compute neurons in Layer 2 (or hidden layer)
A2 = sigmoid( Theta1 * X'); % 25*401 by 401 * 5000

% Add a(0) = 1 for hidden layer
A2 = [ ones(1,m) ; A2 ];

% Computer neurons in layer 3 (Output Layer)
A3 = sigmoid (Theta2 * A2); % 10*26 by 26*5000

temp = zeros(1, size(X, 1)); % 1* 5000
[~,temp] = max(A3, [], 1); % max of each column in A3(10* 5000)

p = temp';


% =========================================================================


end
