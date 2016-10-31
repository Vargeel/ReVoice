import tensorflow as tf

import Model as md


Session = tf.Session()
wavenet_params = {
    "filter_width": 2,
    "sample_rate": 16000,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "residual_channels": 32,
    "dilation_channels": 32,
    "quantization_channels": 256,
    "skip_channels": 512,
    "use_biases": True,
    "scalar_input": False,
    "initial_filter_width": 32
}
net = md.WaveNetModel(
        batch_size=1,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"],
histograms=args.histograms)

logs_path = 'logs/'
writer = tf.train.SummaryWriter(logs_path, Session.graph)