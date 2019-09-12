 % Trace Ratio Problem: min{W'*W = I} Tr(W'*Sw*W)/Tr(W'*Sb*W)
function [W] = solveW(Sb, Sw, d)
% Sb: a matrix to reflects the between-class or global affinity
%     relationship encoded on Graph, Sb = X*Lb*X'
% Sw: a matrix to reflects the within-class or local affinity relationship
%     encoded on Graph, Sw = X*Lw*X'
% d: the number of projected dimension
NITER = 100;
Sb = max(Sb, Sb');
W = eig1(Sb, d, 0, 0);
obj(1) = trace(W'*Sw*W)/trace(W'*Sb*W);

for i = 2 : NITER
    tmp = Sw - trace(W'*Sw*W)/trace(W'*Sb*W)*Sb;
    tmp = max(tmp, tmp');
    W = eig1(tmp, d, 0, 0);
    obj(i) = trace(W'*Sw*W)/trace(W'*Sb*W);
    if abs(obj(i) - obj(i-1)) < 1e-8
        break;
    end;
end
    
    