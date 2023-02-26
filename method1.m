function [label_row, label_col, S] = method1(B, c, NITER)
% input:
% B: input bipartite graph
% c: cluster number
% lambda: hype-parameter
% output:
% label_row, label_col: clustering result to row and column repsectively
% S: structured bipartite graph
    lambda = 10;
    zr = 1e-20;
    m = size(B, 2);
    S = B;
    SS = S' * S;
    SS = (SS + SS')/2;
    D = diag(sum(SS, 1));
    Ls  = D - SS;

    % normalized laplacian
    % D2 = diag(1./sqrt(diag(D)));
    % D2(D2==Inf)=0;
    % Ls = diag(ones(size(SS,1), 1)) - D2*SS*D2;

    % best init F
    [F, ~, ~] = eig1(Ls, c, 0);
    n = size(S, 1);
    % loss = zeros(n,10);
    training_loss = zeros(NITER, 1);
    training_acc = zeros(NITER, 1);

    for iter = 1:NITER
        % ||s - (b=lambda*diag(F*F'+2*lambda*F*F'*s))||^2
        V = B' - lambda*diag(F*F');
        reweighted_param = 2*lambda*(S*(F*F'))';
        for i = 1:n
    % ---- re-weighted loop for updating S ----
    %         v = B(i,:)' - lambda*diag(F*F');
            for iter_re_weighted = 1:1
                old = S(i,:);
                S(i,:) = EProjSimplex_new(V(:,i)+reweighted_param(:,i));
                % diff = norm(old-S(i,:), 2);
                % loss(i, iter_re_weighted) = diff;
                % if (diff < 1e-12)
                %     break
                % end
            end
        end
        SS = S' * S;
        
        D = diag(sum(SS, 1));
        Ls  = D - SS;
        % normalized laplacian
    %     D2 = diag(1./sqrt(diag(D)));
    %     D2(D2==Inf)=0;
    %     Ls = diag(ones(size(SS,1), 1)) - D2*SS*D2;
        F_old = F;
        [F, ~, ev] = eig1(Ls, c, 0);
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1));
        
        if fn1 > 1e-8
            lambda = 2*lambda;
        elseif fn2 < 1e-8
            lambda = lambda/2;  F = F_old;
        else
            break
        end
    end
    SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=S; SS0(n+1:end,1:n)=S';
    [clusternum, label]=graphconncomp(SS0);

    label_row = label(1:n)';
    label_col = label(n+1:end);

    % if clusternum ~= c
    %     sprintf('Can not find the correct cluster number: %d', c)
    % end
end
