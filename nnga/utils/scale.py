def scale_features(sample, header, scale_parameters, scale_method):

    if scale_method == 'Standard':
        for i, key in enumerate(header):
            value = float(sample[i].replace(',', '.'))
            mean = scale_parameters[key]['mean']
            stdev = scale_parameters[key]['stdev'] \
                if scale_parameters[key]['stdev'] != 0 else 0.001
            sample[i] = (value - mean) / stdev

    elif scale_method == 'MinMax':
        for i, key in enumerate(header):
            value = float(sample[i].replace(',', '.'))
            min_val = scale_parameters[key]['min']
            max_val = scale_parameters[key]['max']
            diff = (max_val - min_val) \
                if max_val - min_val != 0 else 0.001
            sample[i] = (value - min_val) / diff

    return sample
