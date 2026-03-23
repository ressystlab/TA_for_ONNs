import sys
sys.path.append("../TA_for_ONNs")

import numpy as np
from neuroptica_new.component_layers import MZILayer, OpticalMesh


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# quantization of input
def quantize_fixed_point_unsigned(array, bits = 8, decimal = 4):
    increment = 1/(2 ** decimal)
    max_num = 2 ** (bits - decimal + 1)
    level_list = np.arange(0, max_num, increment)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = find_nearest(level_list, array[i][j])
            array[i][j] = val

    return array

# quantization of input
def quantize_fixed_point_signed(array, bits = 8, decimal = 4):
    increment = 1/(2 ** decimal)
    max_num = 2 ** (bits - decimal + 1)
    offset = max_num // 2
    level_list = np.arange(0, max_num, increment) - offset

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            val = find_nearest(level_list, array[i][j])
            array[i][j] = val

    return array

def get_phase_levels(V_pi = 1.92, V_2pi = 2.73, V_max = 4, b = 8):
    gamma = np.pi / (V_pi ** 2)

    # generate phase levels
    levels = np.floor((V_max / V_2pi) * 2 ** b)
    v_level = V_max / (2 ** b - 1)
    xvals = [v_level * i for i in range(2 ** b)]

    for i in range(len(xvals)):
        if xvals[i] >= V_2pi:
            xvals[i] = 0

    yinterp = gamma * np.array(xvals) ** 2
    return yinterp

def get_phase_levels_with_noise(V_pi = 1.92, V_2pi = 2.73, V_max = 4, b = 8, uncert = 0.1):
    gamma = np.pi / (V_pi ** 2)

    # generate phase levels
    levels = np.floor((V_max / V_2pi) * 2 ** b)
    v_level = 4 / (2 ** b - 1)
    xvals = [v_level * (i + np.random.normal(0, uncert)) for i in range(2 ** b)]


    yinterp = gamma * np.array(xvals) ** 2

    for i in range(len(yinterp)):
        if yinterp[i] >= np.pi * 2:
            yinterp[i] = 0

    return yinterp

def set_phases_quantized(model, phase_levels, is_diamond = False, loss_dB=0, phase_uncert_phi=0, phase_uncert_theta=0):
    phases_all = model.get_all_phases()
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, 'mesh'):
            phases = phases_all[layer_idx]
            layer_idx += 1

            mzi_nums = [int(len(range(start, end+1))/2) for start, end in zip(layer.mzi_limits_lower, layer.mzi_limits_upper)] # get the number of MZIs in this component layer
            layers = []
            if (None, None) in phases:
                phases = [(None, None) for _ in range(sum(mzi_nums))]

            phases_mzi_layer = []
            idx = 0
            for ii in mzi_nums:
                phases_layer = []
                for jj in range(ii):
                    new_phases = (find_nearest(phase_levels, phases[idx][0]), find_nearest(phase_levels, phases[idx][1]))
                    phases_layer.append(new_phases)
                    idx += 1
                phases_mzi_layer.append(phases_layer)

            # create every layer of MZIs with new loss/phase uncerts

            if is_diamond:
                layerCount = 4 # keep track of which layer we are creating
                val = layer.S
            else:
                layerCount = 0 # keep track of which layer we are creating
                val = layer.N
            for start, end, phases in zip(layer.mzi_limits_lower, layer.mzi_limits_upper, phases_mzi_layer):
                thetas = [phase[0] for phase in phases]
                phis = [phase[1] for phase in phases]
                layers.append(MZILayer.from_waveguide_indices(layerCount, val, list(range(start, end + 1)), thetas=thetas, phis=phis,
                                                              phase_uncert_theta=phase_uncert_theta, phase_uncert_phi=phase_uncert_phi, loss_dB=loss_dB))
                layerCount += 1
            layer.mesh = OpticalMesh(val, layers)

    # phases_quantized = model.get_all_phases()
    # model.set_all_phases_uncerts_losses(phases_quantized, phase_uncert_theta=phase_uncert_theta, phase_uncert_phi=phase_uncert_phi, loss_dB=loss_dB)

def set_phases_quantized_custom(model, phase_levels, N, mesh_profile, loss_dB=0, phase_uncert_phi=0, phase_uncert_theta=0):
      """ Sets the phases changes phase uncerts (phi, theta) and changes the loss_dB """
      mzi_nums = [len(mesh_columns)//2 for mesh_columns in mesh_profile]
      layers = []
      phases = []
      phases_all = model.get_all_phases()
      phases_mzi_layer = []

      for phase in phases_all:
          phases_layer = []
          for p in phase:
              new_phases = (find_nearest(phase_levels, p[0]), find_nearest(phase_levels, p[1]))
              phases_layer.append(new_phases)
          phases_mzi_layer.append(phases_layer)
      model.set_all_phases_uncerts_losses(phases_mzi_layer, phase_uncert_theta=phase_uncert_theta, phase_uncert_phi=phase_uncert_phi, loss_dB=loss_dB)

# weight quantization with voltage source instability
def get_phase_levels_with_noise(V_pi = 1.92, V_2pi = 2.73, V_max = 4, b = 8, uncert = 0.1):
    gamma = np.pi / (V_pi ** 2)

    # generate phase levels
    levels = np.floor((V_max / V_2pi) * 2 ** b)
    v_level = 4 / (2 ** b - 1)
    xvals = [v_level * (i + np.random.normal(0, uncert)) for i in range(2 ** b)]

    for i in range(len(xvals)):
        print(xvals[i])
        if xvals[i] >= V_2pi:
            xvals[i] = 0

    yinterp = gamma * np.array(xvals) ** 2
    return yinterp
