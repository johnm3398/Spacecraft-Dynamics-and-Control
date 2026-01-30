function hw4_problem4_regulator_mrp()
    clc; clear; close all;

    % Inertia (principal axes)
    I1 = 100; I2 = 75; I3 = 80;   % kg*m^2
    I = diag([I1 I2 I3]);

    % Gains
    K = 5;                        % N*m
    P = 10;                       % N*m*s  (P*I3x3)
    Pmat = P*eye(3);

    % Initial conditions
    sigma0 = [0.1; 0.2; -0.1];    % MRP
    w0_deg = [30; 10; -20];       % deg/s
    w0 = deg2rad(w0_deg);         % rad/s

    x0 = [sigma0; w0];

    % Sim settings
    tspan = [0 120];
    opts = odeset('RelTol',1e-10,'AbsTol',1e-12);

    % Integrate
    [t, x] = ode45(@(t,x) eom_mrp_rigidbody(t,x,I,K,Pmat), tspan, x0, opts);

    sigma = x(:,1:3);
    omega = x(:,4:6);

    % MRP norm at t = 30 s (interpolate)
    sigma30 = interp1(t, sigma, 30.0).';
    mrp_norm_30 = norm(sigma30);

    fprintf('MRP norm at t = 30 s: %.12f\n', mrp_norm_30);

    % Quick plots
    figure; plot(t, sigma); grid on;
    xlabel('Time [s]'); ylabel('\sigma'); legend('\sigma_1','\sigma_2','\sigma_3');
    title('MRP Components');

    figure; plot(t, vecnorm(sigma,2,2)); grid on;
    xlabel('Time [s]'); ylabel('||\sigma||'); title('MRP Norm');

    figure; plot(t, rad2deg(omega)); grid on;
    xlabel('Time [s]'); ylabel('\omega [deg/s]');
    legend('\omega_1','\omega_2','\omega_3');
    title('Body Rates');
end

% ========================= Dynamics =========================
function xdot = eom_mrp_rigidbody(~, x, I, K, Pmat)
    sigma = x(1:3);
    w     = x(4:6);

    % Shadow set switch (keeps ||sigma|| <= 1)
    s2 = dot(sigma,sigma);
    if s2 > 1
        sigma = -sigma / s2;
    end

    % Control law for regulator case (R == N, sigma_R/N = 0, L = 0)
    % u = -K*sigma - P*w + w x (I*w)
    u = -K*sigma - Pmat*w + cross(w, I*w);

    % Rigid body dynamics: I*w_dot = - w x (I*w) + u
    wdot = I \ ( -cross(w, I*w) + u );

    % MRP kinematics: sigma_dot = 1/4 * B(sigma) * w
    B = Bmat_mrp(sigma);
    sigmadot = 0.25 * B * w;

    xdot = [sigmadot; wdot];
end

function B = Bmat_mrp(sigma)
    s1 = sigma(1); s2 = sigma(2); s3 = sigma(3);
    s_sq = sigma.'*sigma;

    S = [  0  -s3   s2;
          s3   0  -s1;
         -s2  s1   0 ];

    B = (1 - s_sq)*eye(3) + 2*S + 2*(sigma*sigma.');
end
