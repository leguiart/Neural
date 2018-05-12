filename = 'matlab_data.txt';
delimiterOut = ',';
[A,delimiterOut] = importdata(filename);

    figure(1)
    subplot(2,1,1);
    plot(A(1,:), A(2,:))
    grid on; xlabel('N','fontsize',20); ylabel('Funcion original','fontsize',20);
    xlim([-2 2]);
    
    subplot(2,1,2);
    hold on;
    plot(A(1,:), A(3,:), 'r');
    plot(A(1,:), A(4,:), 'g');
    plot(A(1,:), A(5,:), 'b');
    grid on; xlabel('N','fontsize',20); ylabel('it1, it2, it2000','fontsize',20);
    legend('it2000', 'it1', 'it2');
    xlim([-2 2]);
    hold off;
    
    figure(2)
    hold on;
    plot(A(1,:), A(6,:), 'r')
    plot(A(1,:), A(7,:), 'g')
    grid on; xlabel('N','fontsize',20); ylabel('It2000','fontsize',20);
    legend( 'Funcion original', 'Red2-A');
    xlim([-2 2]);
    hold off;
    
    figure(3)
    hold on;
    plot(A(1,:), A(6,:), 'r')
    plot(A(1,:), A(8,:), 'g')
    grid on; xlabel('N','fontsize',20); ylabel('It10','fontsize',20);
    legend( 'Funcion original', 'Red2-B');
    xlim([-2 2]);
    hold off;
    
    figure(4)
    hold on;
    plot(A(1,:), A(6,:), 'r')
    plot(A(1,:), A(9,:), 'g')
    grid on; xlabel('N','fontsize',20); ylabel('It8','fontsize',20);
    legend( 'Funcion original', 'Red2-C');
    xlim([-2 2]);
    hold off;