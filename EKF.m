%   KF Kalman Filter
%
%   I. System
%        x+ = f(x, u, n)
%        y  = h(x) + v
%        
%        x : state vector            - P : covariance matrix
%        u : control vector          
%        n : perturbation vector     - Q : covariance matrix
%        y : meansurement vector
%        v : measurement noise       - R : covariance matrix 
%
%        f() : transition function
%        h() : measurement function
%        H   : measurement matrix
%
%    II. Initialisation 
%        
%        Define f(), and h()
%
%        precise x, P, Q, R
%        
%
%    III. Temporal loop
%        
%        IIIa. Prediction of mean value of x at the arrival of u
%
%            Jacobian computation
%            F_x : Jac. of x+ wrt to state
%            F_u : Jac. of x+ wrt to control
%            F_n : Jac. of x+ wrt to perturbation
%            
%            x+ = f(x, u, n)
%            P+ = F_x * P F_x' + F_n * Q * F_n'
%            
%        IIIb. correction of mean(x) and P at the arrival of y
%           
%            Jacobian computation
%            H : jac. of y wrt to x
%
%            e  = h(x)                 ...(expected measurement)
%            E  = H * P * H'
%            
%            z  =  y - e              ...(innovation/ new information)
%            Z  = R + E
%
%            K  =  P * H' * Z^(-1)    ...(Kalman gain)
%            x+ = x + K * z 
%
%            P+ = P - K * H * P  // P - K * Z * K'  // and Joseph form
%
%        IV. Plot results 
%
%        V.  How to setup KF examples 
%
%            1. Simulate system, get x, u and y trajectories 
%
%            2. Estimate x with the KF. Get x and P trajectories
%
%            3. Plot results.
