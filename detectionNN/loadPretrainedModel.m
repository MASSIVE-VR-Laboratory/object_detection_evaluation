function net = loadPretrainedModel(opts)
%% Function to load autonn/dagnn model (based on the name provided)
% Input Arguments: Arguments set by the user
% Output Arguments: pre-trained network
% Author: Ratnajit Mukherjee, INESCTEC, Portugal, 2017
% Project: HDR4TT, ONR Global

%% Main body
  net = load(opts.modelPath) ; 
  if ~isfield(net, 'forward') % dagnn loader
    net = dagnn.DagNN.loadobj(net) ;
    switch opts.wrapper
      case 'dagnn' 
        net.mode = 'test' ; 
      case 'autonn'
        out = Layer.fromDagNN(net, @extras_autonn_custom_fn) ; 
        net = Net(out{:}) ;
    end
  else % load directly using autonn
    net = Net(net) ;
  end
  fprintf('\n Pre-trained detector loaded');
end
