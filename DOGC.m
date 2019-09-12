 function [W, S, Q, Final_Y, F, B, D, obj] = DOGC(X, Y, param)
    %% input: dataset X, label Y, parameter param
     %  ouput:  projection matrix  W
     %          similarity matrix  S
     %          the value of objective function  obj
     %          continuous cluster indicator matrix  F
     %          discrete cluster indicator matrix  Final_Y
     %          rotation matrix Q 
     
    %% normalize dataset X
    [dim, num] = size(X);
    X0 = X';
    mX0 = mean(X0);
    X1 = X0 - ones(num,1)*mX0;
    scal = 1./sqrt(sum(X1.*X1)+eps);
    scalMat = sparse(diag(scal));
    X = X1*scalMat;
    X = X';
    
   %% initial parameter variables
    c = length(unique(Y));
    
    % randomly initialize Q,B;
    Q = rand(c, c); 
    Q = orth(Q);
    B = rand(dim,c);
    
    % randomly initialize discrete cluster indicator matrix Pre_Y;
    Pre_Y = zeros(num, c);
    for j = 1 : num
       Pre_Y(j,unidrnd(c)) = 1;
    end
    
    % initialize R (its value depends on Pre_Y and B);
    R = Pre_Y - X' * B; 
    
    % initialize D;
    tempD = 0.5 * param.p * (sqrt(sum(R.^2,2) + eps)).^(param.p - 2);
    D = diag(tempD);
    
    % initialize similarity matrix S
    distX = L2_distance_1(X, X);
    [distX1, idx] = sort(distX, 2);
    S = zeros(num);
    
    for i = 1 : num
        di = distX1(i, 2:param.k + 2);
        rr(i) = 0.5 * (param.k * di(param.k + 1) - sum(di(1:param.k))); % according to determination of epsilon
        id = idx(i,2:param.k + 2);
        S(i,id) = (di(param.k + 1) - di)/(param.k*di(param.k + 1) - sum(di(1:param.k)) + eps);
    end;
    
    if param.r <= 0
        param.r = mean(rr);
    end;
    
    lambda = mean(rr); 
        
    % initialize F by eigenvalue decomposition of L0;
    S0 = (S + S')/2;
    D0 = diag(sum(S0));
    L0 = D0 - S0;
    [F, ~, ~] = eig1(L0,c, 0);
    
    % initialize W;
    H  = eye(num) - 1/num * ones(num);
    St = X * H * X';
    M  = X * L0 * X';
    W  = solveW(St, M, param.d);
    
   %% optimization
    NITER = 10; %the number of iterations
    
    for iter = 1 : NITER
        distf = L2_distance_1(F', F');
        distx = L2_distance_1(W' * X, W' * X);
        if iter > 5
           [~, idx] = sort(distx, 2);
        end;
        
    % update D;
    tempD = 0.5*param.p * (sqrt(sum((Pre_Y - X'*B).^2,2) + eps)).^(param.p - 2);
    D     = diag(tempD);
        
    % update similarity matrix S by adaptive learning;  
    S = zeros(num);
    for i = 1 : num
        idxa0 = idx(i,2:param.k + 1);
        dfi   = distf(i,idxa0);
        dxi   = distx(i,idxa0);
        ad    = -(dxi + lambda * dfi) / (2 * param.r);
        S(i,idxa0) = EProjSimplex_new(ad);
    end; 

    % update L,F;
    S  = (S + S')/2;
    D0 = diag(sum(S));
    L  = D0 -S;
    F  = GPI(L,param.alpha * Pre_Y * Q',1);
        
    % update projective matrix W;
    M = X *L*X';
    W = solveW(St, M, param.d);
      
    % update ratation matrix Q;
    Q = GPI(param.alpha * (F' * F), param.alpha * F' * Pre_Y,1);
      
    % update B;
    B = inv(X * D * X' + param.gamma * eye(dim, dim)) * X * D * Pre_Y;
      
    % update Pre_Y (the final cluster label);
    Pre_Y = zeros(num, c);
    P = param.alpha * F * Q + param.beta * D * X' * B;
    for i = 1:num
        [~,I1] = max(P(i,:));
        Pre_Y(i,I1) = 1;
        Final_Y(i) = I1;
    end
    Final_Y = Final_Y';
      
    % calculate the objective function
    obj(iter) = trace(W' * X * L * X' * W)/trace(W' * St* W)+2 * lambda * trace(F' * L * F) + param.alpha * trace((Pre_Y' - Q' * F') * (Pre_Y - F * Q)) + param.beta * trace((Pre_Y' - B' * X) * D * (Pre_Y - X' * B)) + param.gamma * trace(B' *B);
    end;




