
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

    function [xo, XO_x, XO_u, XO_n] = ex02_f(x, u, n, dt)

        px = x(1);
        py = x(2);
        vx = x(3);
        vy = x(4);
        ax = u(1);
        ay = u(2);
        nx = n(1);
        ny = n(2);
        
        px = px + vx * dt;
        py = py + vy * dt;
        vx = vx + ax * dt + nx;
        vy = vy + ay * dt + ny;

        xo = [px;py;vx;vy];

        if nargout > 1 % we want jacobians 
            
            % transition jacobians 
            XO_x = [...
                    1 0 dt 0 
                    0 1 0 dt
                    0 0 1 0
                    0 0 0 1];

            % control jacobians
            XO_u = [...
                    0 0
                    0 0
                    dt 0
                    0 dt];

            % perturbation Jacobias
            XO_n = [...
                    0 0
                    0 0
                    1 0
                    0 1];

        end
    end

