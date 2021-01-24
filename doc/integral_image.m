% I = [1 7 4 2 9
%      7 2 3 8 2
%      1 8 7 9 1
%      3 2 3 1 5
%      2 9 5 6 6]

% I = magic(6)

I = randi([1, 9], [6,6])

J = integralImage(I)