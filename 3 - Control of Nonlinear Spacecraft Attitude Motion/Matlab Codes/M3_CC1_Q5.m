function hw4_q5_tracking_mrp_fixed()
    clc; clear; close all;

    % Inertia (principal)
    I = diag([100, 75, 80]);       % kg*m^2

    % Gains
    K = 5;                          % N*m
    P = 10*eye(3);                  % N*m*s
    L = zeros(3,1);

    % Reference frequency
    f = 0.05;                       % rad/s

    % Initial conditions
    sigmaBN0 = [0.1; 0.2; -0.1];
    wBN0 = deg2rad([30; 10; -20]);  % rad/s, expressed in B
    x0 = [sigmaBN0; wBN0];

    % Integrate
    tspan = [0 120];
    opts = odeset('RelTol',1e-10,'AbsTol',1e-12);
    [t, x] = ode45(@(t,x) eom_q5(t,x,I,K,P,L,f), tspan, x0, opts);

    % Post-process sigma_B/R
    sigBR = zeros(length(t),3);
    for k = 1:length(t)
        sigBN = shadow_mrp(x(k,1:3).');
        sigRN = shadow_mrp(sigma_ref(t(k), f));
        sigBR(k,:) = shadow_mrp(mrp_compose(sigBN, mrp_inverse(sigRN))).';
    end

    % value at 40s
    sigBR_40 = interp1(t, sigBR, 40.0).';
    val = norm(sigBR_40);

    fprintf('Q5: ||sigma_{B/R}|| at t=40 s = %.15f\n', val);

    figure; plot(t, vecnorm(sigBR,2,2)); grid on;
    xlabel('Time [s]'); ylabel('||\sigma_{B/R}||'); title('Tracking Error Norm');
end

% -------------------- Dynamics (Q5) --------------------
function xdot = eom_q5(t, x, I, K, P, L, f)
    sigBN = shadow_mrp(x(1:3));
    wBN_B = x(4:6);                 % omega_B/N expressed in B

    % Reference attitude sigma_R/N(t)
    sigRN = shadow_mrp(sigma_ref(t, f));
    sigRN_dot = sigma_ref_dot(t, f);

    % Compute omega_R/N expressed in R using MRP kinematics:
    % sigdot = 1/4 * B(sig) * omega  => omega = 4 * B^{-1} * sigdot
    wRN_R = 4 * (Bmat_mrp(sigRN) \ sigRN_dot);

    % Attitude error sigma_B/R = sigma_B/N o (sigma_R/N)^{-1}
    sigBR = shadow_mrp(mrp_compose(sigBN, mrp_inverse(sigRN)));

    % DCM C_B/R from sigma_B/R
    C_BR = mrp_to_dcm(sigBR);

    % Rate error omega_B/R expressed in B:
    % omega_B/R^B = omega_B/N^B - C_B/R * omega_R/N^R
    wBR_B = wBN_B - C_BR*wRN_R;

    % Controller for this homework (consistent with the stabilizing form)
    % Use the same cancellation structure: + (wBN x I*wBN)
    u = -K*sigBR - P*wBR_B + cross(wBN_B, I*wBN_B) - L;

    % Rigid body dynamics: I*w_dot + w x (I*w) = u
    wdot = I \ (u - cross(wBN_B, I*wBN_B));

    % Kinematics for sigma_B/N
    sigdot = 0.25 * Bmat_mrp(sigBN) * wBN_B;

    xdot = [sigdot; wdot];
end

% -------------------- Reference --------------------
function sig = sigma_ref(t, f)
    sig = [0.2*sin(f*t); 0.3*cos(f*t); -0.3*sin(f*t)];
end

function sigdot = sigma_ref_dot(t, f)
    sigdot = [0.2*f*cos(f*t); -0.3*f*sin(f*t); -0.3*f*cos(f*t)];
end

% -------------------- MRP utilities --------------------
function sig = shadow_mrp(sig)
    s2 = dot(sig,sig);
    if s2 > 1
        sig = -sig / s2;
    end
end

function siginv = mrp_inverse(sig)
    siginv = -sig;
end

function sigAC = mrp_compose(sigAB, sigBC)
    a = sigAB; 
    b = sigBC;
    a2 = dot(a,a); 
    b2 = dot(b,b);

    denom = 1 + a2*b2 - 2*dot(a,b);

    % Correct cross-term sign for Schaub MRP composition:
    % -2*(b x a)  == +2*(a x b)
    sigAC = ( (1 - b2)*a + (1 - a2)*b + 2*cross(a,b) ) / denom;
end


function B = Bmat_mrp(sig)
    s2 = dot(sig,sig);
    S = skew(sig);
    B = (1 - s2)*eye(3) + 2*S + 2*(sig*sig.');
end

function S = skew(v)
    S = [  0   -v(3)  v(2);
          v(3)  0   -v(1);
         -v(2) v(1)   0  ];
end

function C = mrp_to_dcm(sig)
    sig = shadow_mrp(sig);
    s2 = dot(sig,sig);
    S = skew(sig);
    C = eye(3) + (8*(S*S) - 4*(1 - s2)*S) / (1 + s2)^2;
end
