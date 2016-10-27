clear all;
clc;
load TestScore
load Passed
TS=TestScore;
P=Passed;
[m,n] = size(TestScore);
TS = [ones(m, 1), TS]; 
Pass   = find(P==1);
Reject = find(P==0);
figure(1)
plot(TS(Pass, 2), TS(Pass,3), '+')
hold on
plot(TS(Reject, 2), TS(Reject, 3), 'o')
% ********************************************************
hypothesis = @(z) 1.0 ./ (1.0 + exp(-z));
CostFunction =@(h) (1/m)*sum(-P.*log(h) - (1-P).*log(1-h));
Gradient=@(h) (1/m).*TS' * (h-P);
Hessian=@(h) (1/m).*TS' * diag(h) * diag(1-h) * TS;
theta = zeros(n+1, 1); % starting with the initial value of θ = 0.⃗
Iteration = 15; % Newton’s method often converges in 5-15 iterations.
J = zeros(Iteration, 1); % in start time cost=0.
% **********************************************************
for i = 1:Iteration
    k(i)=i;
    z = TS * theta;
    h=hypothesis(z);
    g =Gradient(h);
    H =Hessian(h);
    J(i) = CostFunction(h);
    theta = theta - inv(H)*g; %update theta
end
% **********************************************************
X_value = [min(TS(:,2)),max(TS(:,2))];
Y_value = (-1./theta(3)).*(theta(2).*X_value +theta(1));
plot(X_value,Y_value); %Plotting the decision boundary
legend('Admitted', 'Not admitted', 'Decision Boundary')
% ************************************************
figure(2)
plot(k,J','o--','MarkerFaceColor','g');
xlabel('Iteration'); ylabel('J');
i=0;
while(1)
    i=i+1;
    if J(i)-J(i+1) < 0.001
        OptimomIttiration=k(i)
    break;
    end
end
% ****************************************************
% ************************************************
score1=input('first exam score = ');
score2=input('second exam score = ');
message=['probability that a student with a score of ',num2str(score1),' on Exam 1 and a score of ',num2str(score2),'on Exam 2 will not be admitted' ];
disp(message)
probability = 1 - hypothesis([1, score1, score2]*theta)