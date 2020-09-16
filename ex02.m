    close all;
    clear all;
    % define system
    % x = [px, py, vx, vy]'
    % y = [range, bearing]'
    % n = [nx, ny]'
    % u = [ux, uy]'
    % r = [rd, ra]'
    % px+ = px + vx *dt
    % py+ = py + vy * dt
    % vx+ = vx + ax*dt + nx
    % vy+ = vy + ay*dt + ny
    % 
    % range = sqrt(px^2 + py^2) + rd
    % bearing = atan2(py, px) + ra
    dt = 0.1;
    
    Q = diag([ 0.01, 0.01].^2);
    R = diag([ 0.1, 1*pi/180].^2);



    
    % simulated variable

    X = [2, 1, -1, 1]';

    % estimated variables
    
    x = [3, 3, 0, 0]';
    P = diag([1, 2, 1, 1].^2);

    % trajectories 
    tt = 0:dt:4;
    XX = zeros(4, size(tt, 2));
    xx = zeros(4, size(tt, 2));
    yy = zeros(2, size(tt, 2));
    pp = zeros(4, size(tt, 2));

    % perturbation levels 
    q = sqrt(diag(Q)) / 2;
    r = sqrt(diag(R)) / 2;
    % start loop
    
    i = 1;
    for t = tt
        
        % read control

        u = [0, 0]';
        % simulate

        n = q .* randn(2, 1);
        X = ex02_f(X, u, n, dt);
        v = r .* randn(2, 1);
        y = ex02_h(X) + v;

        % estimate prediction

        [x, F_x, F_u, F_n] = ex02_f(x, u, zeros(2, 1), dt);
        P = F_x * P * F_x' + F_n * Q * F_n';

        % correction step
        [e, H] = ex02_h(x);
        E = H * P * H';

        z = y - e;
        Z = R + E;

        K = P * H' * Z^-1;

        x = x + K*z ;
        P = P - K * H * P;

        % collect data
        %XX(:, i) = X;
        %xx(:, i) = x;
        %yy(:, i) = y;
        %PP(:, i) = diag(P);

        % update the index

        i = i + 1;

        %plots
        plot(X(1), X(2), '*r')
        axis([-4 6 0 4])
        hold on 
        plot(x(1), x(2), '*b')
        hold off
        pause(0.5)
        drawnow
    end
    
    % plot
    %plot(tt,xx,tt,yy,tt,sqrt(PP),XX)
    %legend('state','measurement','sigma','actual')
