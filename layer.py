import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
import numpy as np
import math

debug = False
from keras import initializers


def debug_print(*objects):
    if debug:
        print(*objects)


class MusicSinc1D(Layer):
    def __init__(self, N_filt, Filt_dim, fs, **kwargs):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(MusicSinc1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name="filt_b1",
            shape=(self.N_filt,),
            initializer="he_uniform",
            trainable=True,
        )
        self.filt_band = self.add_weight(
            name="filt_band",
            shape=(self.N_filt,),
            initializer="he_uniform",
            trainable=True,
        )

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + (self.fs / 2) / 700)  # Convert Hz to Mel
        mel_points = np.linspace(
            low_freq_mel, high_freq_mel, self.N_filt
        )  # Equally spaced in Mel scale
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1 / self.freq_scale, (b2 - b1) / self.freq_scale])

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        self.filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        self.filt_end_freq = self.filt_beg_freq + (
            K.abs(self.filt_band) + min_band / self.freq_scale
        )

        # Filter window (hamming).
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")
        self.window = K.variable(window)
        debug_print("  window", self.window.shape)

        # TODO what is this?
        t_right_linspace = np.linspace(
            1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2)
        )
        self.t_right = K.variable(t_right_linspace / self.fs)
        debug_print("  t_right", self.t_right)

        super(MusicSinc1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        debug_print("call")
        # filters = K.zeros(shape=(N_filt, Filt_dim))
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (
            K.abs(self.filt_band) + min_band / self.freq_scale
        )

        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")
        window = K.variable(window)

        t_right_linspace = np.linspace(
            1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2)
        )
        t_right = K.variable(t_right_linspace / self.fs)

        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = (
                2
                * self.filt_beg_freq[i]
                * sinc(self.filt_beg_freq[i] * self.freq_scale, self.t_right)
            )
            low_pass2 = (
                2
                * self.filt_end_freq[i]
                * sinc(self.filt_end_freq[i] * self.freq_scale, self.t_right)
            )
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * self.window)
        filters = K.stack(output_list)  # (80, 251)
        filters = K.transpose(filters)  # (251, 80)
        filters = K.reshape(
            filters, (self.Filt_dim, 1, self.N_filt)
        )  # (251,1,80) in TF: (filter_width, in_channels, out_channels) in
        # PyTorch (out_channels, in_channels, filter_width)

        """Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC", or [batch, 
        in_channels, in_width] if data_format is "NCW", and a filter / kernel tensor of shape [filter_width, 
        in_channels, out_channels], this op reshapes the arguments to pass them to conv2d to perform the equivalent 
        convolution operation. Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, 
        if data_format does not start with "NC", a tensor of shape [batch, in_width, in_channels] is reshaped to [
        batch, 1, in_width, in_channels], and the filter is reshaped to [1, filter_width, in_channels, out_channels]. 
        The result is then reshaped back to [batch, out_width, out_channels] (where out_width is a function of the 
        stride and padding as in conv2d) and returned to the caller. """

        # Do the convolution.
        debug_print("call")
        debug_print("  x", x)
        debug_print("  filters", filters)
        out = K.conv1d(x, kernel=filters)
        debug_print("  out", out)

        return out

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1], self.Filt_dim, padding="valid", stride=1, dilation=1
        )
        return (input_shape[0],) + (new_size,) + (self.N_filt,)


def sinc(band, t_right):
    y_right = K.sin(2 * band * t_right) / (2 * band * t_right)
    # y_left = flip(y_right, 0) TODO remove if useless
    y_left = K.reverse(y_right, 0)
    var = K.variable(np.array([1]))
    y = K.concatenate([y_left, var, y_right])
    return y
