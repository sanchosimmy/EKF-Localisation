    % define system
    % x+ = x + u * dt + n 
    % y = x + v

    dt = 1;

    F_x = 1;
    F_u = dt;
    F_n = 1;
    H = 1;

    Q = 10;
    R = 100;

    
    % simulated variable

    X = 7;
    u = 1;

    % estimated variables
    
    x = 0;
    P = 10e4;

    % trajectories 
    tt = 0:dt:500;
    XX = zeros(1, size(tt, 2));
    xx = zeros(1, size(tt, 2));
    yy = zeros(1, size(tt, 2));
    pp = zeros(1, size(tt, 2));

    % perturbation levels 
    q = sqrt(Q);
    r = sqrt(R);
    % start loop
    
    i = 1;
    for t = tt
        
        % simulate
        n = q * randn;
        X = F_x * X + F_u * u + F_n * n ;
        v = r * randn;
        y = H * X + v;

        % estimate

        x = F_x * x + + F_u * u + F_n;
        P = F_x * P * F_x' + F_n * Q * F_n';

        % correction step
        e = H * x;
        E = H * P * H';

        z = y - e;
        Z = R + E;

        K = P * H' * Z^-1;

        x = x + K*z ;
        P = P - K * H * P;

        % collect data
        XX(:, i) = X;
        xx(:, i) = x;
        yy(:, i) = y;
        PP(:, i) = diag(P);

        % update the index

        i = i + 1;
    end
    
    % plot
    plot(tt,xx,tt,yy,tt,sqrt(PP),XX)
    legend('state','measurement','sigma','actual')
