def scale_features(sample, header, scale_parameters, scale_method):
    """
        Parameters
        ----------
        sample : list
            List containing all to features to be scaled

        header : list
            Features name

        scale_parameters : dict
            Dict containing the scale parameters
                - Mean
                - Stdev
                - Min
                - Max

        scale_method : str
            Method to be used to scale the data
                - Standard
                - MinMax

        Returns
        -------
            sample scaled

    """
    scaled = []
    if scale_method == "Standard":
        for i, key in enumerate(header):
            mean = scale_parameters[key]["mean"]
            stdev = (
                scale_parameters[key]["stdev"]
                if scale_parameters[key]["stdev"] != 0
                else 0.001
            )
            scaled.append((sample[i] - mean) / stdev)

    elif scale_method == "MinMax":
        for i, key in enumerate(header):
            min_val = scale_parameters[key]["min"]
            max_val = scale_parameters[key]["max"]
            diff = (max_val - min_val) if max_val - min_val != 0 else 0.001
            scaled.append((sample[i] - min_val) / diff)

    else:
        return sample

    return scaled
