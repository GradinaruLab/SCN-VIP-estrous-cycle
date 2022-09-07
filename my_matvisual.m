%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Matrix Visualization with MATLAB Implementation    %
%                                                      %
% Author: Ph.D. Eng. Hristo Zhivomirov        12/02/18 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% edited Anat Kahan June 2022

function matvisual(A, varargin,myLabel)

% function: matvisual(A, varargin)
% A - m-by-n matrix to be visualized with up to 3 pages
% varargin - type 'annotation' in the place of varargin if one want to
%            annotate the matrix plot (x-label, y-label, etc.)

% check the input
if ~isreal(A) || isempty(A) || ischar(A) || ndims(A) > 3
    errordlg('The data array is not suitable for visualization!', ...
             'Error!', 'modal')
    return
end

% determine the matrix size
[M, N, P] = size(A);
% Create figure
fh = figure();
ax = axes(fh);
% loop through the matrix pages
for p = 1:P
    
    % prepare page-by-page visualization
    if P > 1, subplot(1, P, p), end 
    
    % visualize the matrix
    himg = imagesc(A(:, :, p),[50 100]);
    
    grid on
    
    % annotation
    if strcmp(varargin, 'annotation')       
        % x-label, y-label, x-ticks, y-ticks, title
        set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
        xlabel('Column number')
        ylabel('Row number')
        if P > 1, title(['Matrix page ' num2str(p)]), end
        if M <= 50, set(gca, 'YTick', 1:M), end 
        if N <= 50, set(gca, 'XTick', 1:N), end   
        tickvalues = 1:M;
        set(ax,'YTick',tickvalues,'YTickLabel',myLabel(end:-1:1))
        ax.XAxis.TickLabelRotation = 90;
        set(ax,'XTick',tickvalues,'XTickLabel',myLabel)
        ax.XAxis.TickLabelRotation = 90;
    end
        % values labeling
        for m = 1:M
            for n = 1:N
                text(n, m, num2str(A(m, n, p), 3), ...
                    'FontName', 'Times New Roman', ...
                    'FontSize', round(6 + 50./sqrt(M.*N)), ...
                    'HorizontalAlignment', 'center', ...
                    'Rotation', 45)
            end
        end
            
    % set the datatip UpdateFcn
    cursorMode = datacursormode(gcf);
    set(cursorMode, 'UpdateFcn', {@datatiptxt, himg})
    colorbar
    colormap('gray');
    himg.AlphaData = .75;
    axis(ax,'equal')
    axis(ax,'tight')
end

end

function text_to_display = datatiptxt(~, hDatatip, himg)

% determine the current datatip position
pos = get(hDatatip, 'Position');

% form the datatip label
text_to_display = {['Row: ' num2str(pos(2))], ...
                   ['Column: ' num2str(pos(1))], ...
                   ['Value: ' num2str(himg.CData(pos(2), pos(1)))]};

end