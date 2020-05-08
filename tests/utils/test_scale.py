from nnga.utils.scale import scale_features

sample = [0, 1]
header = ["Feature_A", "Feature_B"]
scale_parameters = {
    "Feature_A": {"min": 0, "max": 2, "mean": 0, "stdev": 1},
    "Feature_B": {"min": 0, "max": 10, "mean": 5, "stdev": 10},
}


def test_standard_scale():
    scaled = scale_features(sample, header, scale_parameters, "Standard")
    assert scaled == [0.0, -0.4]


def test_minmax_scale():
    scaled = scale_features(sample, header, scale_parameters, "MinMax")
    assert scaled == [0.0, 0.1]


def test_wrong_scale():
    scaled = scale_features(sample, header, scale_parameters, "wrong")
    assert scaled == [0, 1]
