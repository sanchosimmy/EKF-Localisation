%   KF Kalman Filter
%
%   I. System
%       x+ = F_x * x + F_u * u + F_n * n
%        y  = H * x + v
%        
%        x : state vector            - P : covariance matrix
%        u : control vector          
%        n : perturbation vector     - Q : covariance matrix
%        y : meansurement vector
%        v : measurement noise       - R : covariance matrix 
%
%        F_x : transistion matrix
%        F_u : control matrix
%        F_n : perturbation matrix
%        H   : measurement matrix
%
%    II. Initialisation 
%        
%        Define F_x, F_u, F_n and H
%
%        precise x, P, Q, R
%        
%
%    III. Temporal loop
%        
%        IIIa. Prediction of mean value of x at the arrival of u
%
%            x+ = F_x * x + F_u * u + ......(F_n * 0)
%            P+ = F_x * P F_x' + F_n * Q * F_n'
%            
%        IIIb. correction of mean(x) and P at the arrival of y
%
%            e  = H * x               ...(expected measurement)
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
